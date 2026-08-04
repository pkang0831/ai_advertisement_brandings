import CryptoKit
import Foundation
import LocalAuthentication
import Security

private let service = "com.devenira.rina.promotion.secure-enclave"
private let account = "production-p256-v1"

enum SignerError: Error, CustomStringConvertible {
    case message(String)

    var description: String {
        switch self {
        case .message(let value): return value
        }
    }
}

func keychainQuery() -> [String: Any] {
    [
        kSecClass as String: kSecClassGenericPassword,
        kSecAttrService as String: service,
        kSecAttrAccount as String: account,
    ]
}

func readKeyHandle() throws -> Data {
    var query = keychainQuery()
    query[kSecReturnData as String] = true
    query[kSecMatchLimit as String] = kSecMatchLimitOne
    var result: CFTypeRef?
    let status = SecItemCopyMatching(query as CFDictionary, &result)
    guard status == errSecSuccess, let data = result as? Data else {
        throw SignerError.message("secure_enclave_key_not_enrolled")
    }
    return data
}

func keyHandleMetadataExists() -> Bool {
    var query = keychainQuery()
    query[kSecReturnAttributes as String] = true
    query[kSecMatchLimit as String] = kSecMatchLimitOne
    var result: CFTypeRef?
    return SecItemCopyMatching(query as CFDictionary, &result) == errSecSuccess
}

func storeKeyHandle(_ data: Data) throws {
    var query = keychainQuery()
    SecItemDelete(query as CFDictionary)
    query[kSecValueData as String] = data
    query[kSecAttrAccessible as String] = kSecAttrAccessibleWhenUnlockedThisDeviceOnly
    let status = SecItemAdd(query as CFDictionary, nil)
    guard status == errSecSuccess else {
        throw SignerError.message("keychain_store_failed:\(status)")
    }
}

func pemPublicKey(_ der: Data) -> String {
    let encoded = der.base64EncodedString(options: [.lineLength64Characters])
    return "-----BEGIN PUBLIC KEY-----\n\(encoded)\n-----END PUBLIC KEY-----\n"
}

func writeJSON(_ value: [String: Any]) throws {
    let data = try JSONSerialization.data(withJSONObject: value, options: [.sortedKeys])
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write(Data([0x0A]))
}

func requireEnrollmentPresence() throws {
    let context = LAContext()
    context.localizedReason = "Enroll the Rina human-promotion signing key"
    context.localizedFallbackTitle = "Use macOS Password"
    var policyError: NSError?
    guard context.canEvaluatePolicy(
        .deviceOwnerAuthentication,
        error: &policyError
    ) else {
        throw policyError ?? SignerError.message("user_presence_unavailable")
    }
    let semaphore = DispatchSemaphore(value: 0)
    var allowed = false
    var evaluationError: Error?
    context.evaluatePolicy(
        .deviceOwnerAuthentication,
        localizedReason: context.localizedReason
    ) { success, error in
        allowed = success
        evaluationError = error
        semaphore.signal()
    }
    semaphore.wait()
    guard allowed else {
        throw evaluationError ?? SignerError.message("enrollment_presence_denied")
    }
}

func enroll(publicKeyPath: String) throws {
    guard SecureEnclave.isAvailable else {
        throw SignerError.message("secure_enclave_unavailable")
    }
    if (try? readKeyHandle()) != nil {
        throw SignerError.message("already_enrolled")
    }
    try requireEnrollmentPresence()
    var error: Unmanaged<CFError>?
    guard let access = SecAccessControlCreateWithFlags(
        nil,
        kSecAttrAccessibleWhenUnlockedThisDeviceOnly,
        [.privateKeyUsage, .userPresence],
        &error
    ) else {
        throw error?.takeRetainedValue() ?? SignerError.message("access_control_creation_failed")
    }
    let key = try SecureEnclave.P256.Signing.PrivateKey(accessControl: access)
    try storeKeyHandle(key.dataRepresentation)
    let publicDER = key.publicKey.derRepresentation
    let destination = URL(fileURLWithPath: publicKeyPath)
    try FileManager.default.createDirectory(
        at: destination.deletingLastPathComponent(),
        withIntermediateDirectories: true
    )
    try pemPublicKey(publicDER).write(to: destination, atomically: true, encoding: .utf8)
    chmod(destination.path, 0o644)
    try writeJSON([
        "status": "enrolled",
        "secure_enclave": true,
        "user_presence": true,
        "private_key_exportable": false,
        "public_key_path": destination.path,
    ])
}

func readiness(publicKeyPath: String) throws {
    // Metadata lookup is intentionally used here. Reading the encrypted key
    // handle can invoke Keychain UI after a helper rebuild; signing is the only
    // readiness transition allowed to request user presence.
    let enrolled = keyHandleMetadataExists()
    let publicKeyExists = FileManager.default.fileExists(atPath: publicKeyPath)
    var publicKeyValid = false
    var publicKeySHA256: String?
    var probeError: String?
    if publicKeyExists {
        do {
            let pem = try String(
                contentsOfFile: publicKeyPath,
                encoding: .utf8
            )
            let fileKey = try P256.Signing.PublicKey(pemRepresentation: pem)
            publicKeyValid = true
            publicKeySHA256 = SHA256.hash(
                data: fileKey.derRepresentation
            ).map { String(format: "%02x", $0) }.joined()
        } catch {
            probeError = String(describing: error)
        }
    }
    let publicKeySHA256JSON: Any = (
        publicKeySHA256 == nil ? NSNull() : publicKeySHA256!
    )
    let probeErrorJSON: Any = (
        probeError == nil ? NSNull() : probeError!
    )
    try writeJSON([
        "secure_enclave_available": SecureEnclave.isAvailable,
        "enrolled": enrolled,
        "public_key_exists": publicKeyExists,
        "public_key_valid": publicKeyValid,
        "public_key_sha256": publicKeySHA256JSON,
        "non_signing_probe": true,
        "key_pair_match": "pending_live_signature_verification",
        "probe_error": probeErrorJSON,
        "production_ready": (
            SecureEnclave.isAvailable
                && enrolled
                && publicKeyExists
                && publicKeyValid
        ),
        "user_presence_policy": "SecAccessControl.userPresence",
        "private_key_exportable": false,
    ])
}

func sign(payloadPath: String, reason: String) throws {
    guard SecureEnclave.isAvailable else {
        throw SignerError.message("secure_enclave_unavailable")
    }
    let payload = try Data(contentsOf: URL(fileURLWithPath: payloadPath))
    let context = LAContext()
    context.localizedReason = reason
    context.localizedFallbackTitle = "Use macOS Password"
    let key = try SecureEnclave.P256.Signing.PrivateKey(
        dataRepresentation: try readKeyHandle(),
        authenticationContext: context
    )
    let signature = try key.signature(for: payload)
    try writeJSON([
        "algorithm": "ECDSA_P256_SHA256",
        "signature_der_base64": signature.derRepresentation.base64EncodedString(),
        "public_key_der_base64": key.publicKey.derRepresentation.base64EncodedString(),
        "user_presence_enforced": true,
    ])
}

func argument(_ name: String, in args: [String]) throws -> String {
    guard let index = args.firstIndex(of: name), index + 1 < args.count else {
        throw SignerError.message("missing_argument:\(name)")
    }
    return args[index + 1]
}

do {
    let args = Array(CommandLine.arguments.dropFirst())
    guard let command = args.first else {
        throw SignerError.message("usage: readiness|enroll|sign")
    }
    switch command {
    case "readiness":
        try readiness(publicKeyPath: argument("--public-key", in: args))
    case "enroll":
        try enroll(publicKeyPath: argument("--public-key", in: args))
    case "sign":
        try sign(
            payloadPath: argument("--payload", in: args),
            reason: argument("--reason", in: args)
        )
    default:
        throw SignerError.message("unknown_command:\(command)")
    }
} catch {
    let message = String(describing: error)
    let data = try JSONSerialization.data(
        withJSONObject: ["status": "blocked", "error": message],
        options: [.sortedKeys]
    )
    FileHandle.standardError.write(data)
    FileHandle.standardError.write(Data([0x0A]))
    exit(2)
}
