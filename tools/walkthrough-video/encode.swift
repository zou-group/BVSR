// Encode a folder of JPEG frames into an H.264 MP4 with AVFoundation.
// usage: encode <framesDir> <out.mp4> <fps> <bitrate>
import AVFoundation
import CoreGraphics
import ImageIO
import Foundation

let args = CommandLine.arguments
let dir = args[1], out = args[2]
let fps = Int32(args[3])!, bitrate = Int(args[4])!
let files = try FileManager.default.contentsOfDirectory(atPath: dir).filter { $0.hasSuffix(".jpg") }.sorted()
let W = 1920, H = 1080
let url = URL(fileURLWithPath: out)
try? FileManager.default.removeItem(at: url)
let writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
let settings: [String: Any] = [
  AVVideoCodecKey: AVVideoCodecType.h264,
  AVVideoWidthKey: W, AVVideoHeightKey: H,
  AVVideoColorPropertiesKey: [
    AVVideoColorPrimariesKey: AVVideoColorPrimaries_ITU_R_709_2,
    AVVideoTransferFunctionKey: AVVideoTransferFunction_ITU_R_709_2,
    AVVideoYCbCrMatrixKey: AVVideoYCbCrMatrix_ITU_R_709_2],
  AVVideoCompressionPropertiesKey: [
    AVVideoAverageBitRateKey: bitrate,
    AVVideoProfileLevelKey: AVVideoProfileLevelH264HighAutoLevel,
    AVVideoMaxKeyFrameIntervalKey: 60,
    AVVideoExpectedSourceFrameRateKey: Int(fps)]
]
let input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
input.expectsMediaDataInRealTime = false
let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: [
  kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32ARGB,
  kCVPixelBufferWidthKey as String: W, kCVPixelBufferHeightKey as String: H])
writer.shouldOptimizeForNetworkUse = true
writer.add(input)
guard writer.startWriting() else { print("cannot start: \(String(describing: writer.error))"); exit(1) }
writer.startSession(atSourceTime: .zero)
let space = CGColorSpace(name: CGColorSpace.sRGB)!
for (i, f) in files.enumerated() {
  autoreleasepool {
    let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: dir + "/" + f) as CFURL, nil)!
    let img = CGImageSourceCreateImageAtIndex(src, 0, nil)!
    var pbOpt: CVPixelBuffer?
    CVPixelBufferPoolCreatePixelBuffer(nil, adaptor.pixelBufferPool!, &pbOpt)
    let pb = pbOpt!
    CVPixelBufferLockBaseAddress(pb, [])
    let ctx = CGContext(data: CVPixelBufferGetBaseAddress(pb), width: W, height: H, bitsPerComponent: 8,
                        bytesPerRow: CVPixelBufferGetBytesPerRow(pb), space: space,
                        bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue)!
    ctx.draw(img, in: CGRect(x: 0, y: 0, width: W, height: H))
    CVPixelBufferUnlockBaseAddress(pb, [])
    while !input.isReadyForMoreMediaData { usleep(2000) }
    if !adaptor.append(pb, withPresentationTime: CMTime(value: Int64(i), timescale: fps)) {
      print("append failed at \(i): \(String(describing: writer.error))"); exit(1)
    }
  }
}
input.markAsFinished()
let sem = DispatchSemaphore(value: 0)
writer.finishWriting { sem.signal() }
sem.wait()
print("wrote \(files.count) frames -> \(out) status=\(writer.status.rawValue)")
