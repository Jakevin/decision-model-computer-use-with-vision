import CoreGraphics
import Foundation
import ImageIO
import Vision

guard CommandLine.arguments.count >= 2 else {
    fputs("Usage: vision_ocr <image_path>\n", stderr)
    exit(2)
}

let imagePath = CommandLine.arguments[1]
let imageUrl = URL(fileURLWithPath: imagePath)

guard let imageSource = CGImageSourceCreateWithURL(imageUrl as CFURL, nil),
      let cgImage = CGImageSourceCreateImageAtIndex(imageSource, 0, nil)
else {
    fputs("Failed to load image at \(imagePath)\n", stderr)
    exit(1)
}

let request = VNRecognizeTextRequest()
request.recognitionLevel = .accurate
request.recognitionLanguages = ["zh-Hant", "en-US"]
request.usesLanguageCorrection = true

let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
do {
    try handler.perform([request])
} catch {
    fputs("Vision OCR failed: \(error)\n", stderr)
    exit(1)
}

let observations = request.results ?? []
for obs in observations {
    guard let candidate = obs.topCandidates(1).first else {
        continue
    }
    let bbox = obs.boundingBox
    let obj: [String: Any] = [
        "text": candidate.string,
        "confidence": Double(candidate.confidence),
        "x": Double(bbox.origin.x),
        "y": Double(bbox.origin.y),
        "w": Double(bbox.size.width),
        "h": Double(bbox.size.height),
    ]
    if let data = try? JSONSerialization.data(withJSONObject: obj, options: []),
       let line = String(data: data, encoding: .utf8)
    {
        print(line)
    }
}
