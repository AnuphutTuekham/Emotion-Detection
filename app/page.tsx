// import Image from "next/image";

// export default function Home() {
//   return (
//     <div className="flex min-h-screen items-center justify-center bg-zinc-50 font-sans dark:bg-black">
//       <main className="flex min-h-screen w-full max-w-3xl flex-col items-center justify-between py-32 px-16 bg-white dark:bg-black sm:items-start">
//         <Image
//           className="dark:invert"
//           src="/next.svg"
//           alt="Next.js logo"
//           width={100}
//           height={20}
//           priority
//         />
//         <div className="flex flex-col items-center gap-6 text-center sm:items-start sm:text-left">
//           <h1 className="max-w-xs text-3xl font-semibold leading-10 tracking-tight text-black dark:text-zinc-50">
//             To get started, edit the page.tsx file.
//           </h1>
//           <p className="max-w-md text-lg leading-8 text-zinc-600 dark:text-zinc-400">
//             Looking for a starting point or more instructions? Head over to{" "}
//             <a
//               href="https://vercel.com/templates?framework=next.js&utm_source=create-next-app&utm_medium=appdir-template-tw&utm_campaign=create-next-app"
//               className="font-medium text-zinc-950 dark:text-zinc-50"
//             >
//               Templates
//             </a>{" "}
//             or the{" "}
//             <a
//               href="https://nextjs.org/learn?utm_source=create-next-app&utm_medium=appdir-template-tw&utm_campaign=create-next-app"
//               className="font-medium text-zinc-950 dark:text-zinc-50"
//             >
//               Learning
//             </a>{" "}
//             center.
//           </p>
//         </div>
//         <div className="flex flex-col gap-4 text-base font-medium sm:flex-row">
//           <a
//             className="flex h-12 w-full items-center justify-center gap-2 rounded-full bg-foreground px-5 text-background transition-colors hover:bg-[#383838] dark:hover:bg-[#ccc] md:w-[158px]"
//             href="https://vercel.com/new?utm_source=create-next-app&utm_medium=appdir-template-tw&utm_campaign=create-next-app"
//             target="_blank"
//             rel="noopener noreferrer"
//           >
//             <Image
//               className="dark:invert"
//               src="/vercel.svg"
//               alt="Vercel logomark"
//               width={16}
//               height={16}
//             />
//             Deploy Now
//           </a>
//           <a
//             className="flex h-12 w-full items-center justify-center rounded-full border border-solid border-black/[.08] px-5 transition-colors hover:border-transparent hover:bg-black/[.04] dark:border-white/[.145] dark:hover:bg-[#1a1a1a] md:w-[158px]"
//             href="https://nextjs.org/docs?utm_source=create-next-app&utm_medium=appdir-template-tw&utm_campaign=create-next-app"
//             target="_blank"
//             rel="noopener noreferrer"
//           >
//             Documentation
//           </a>
//         </div>
//       </main>
//     </div>
//   );
// }


"use client";
/* eslint-disable @typescript-eslint/no-explicit-any */

import { useEffect, useRef, useState } from "react";
import * as ort from "onnxruntime-web";

type CvType = unknown;

export default function Home() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const [status, setStatus] = useState<string>("ยังไม่เริ่ม");
  const [emotion, setEmotion] = useState<string>("-");
  const [conf, setConf] = useState<number>(0);
  const [isRecording, setIsRecording] = useState<boolean>(false);
  const [fps, setFps] = useState<number>(0);

  const cvRef = useRef<CvType | null>(null);
  const faceCascadeRef = useRef<unknown>(null);
  const sessionRef = useRef<ort.InferenceSession | null>(null);
  const classesRef = useRef<string[] | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const frameCountRef = useRef<number>(0);
  const loopRef = useRef<boolean>(true);

  // Load OpenCV.js
  // async function loadOpenCV() {
  //   if (typeof window === "undefined") return;

  //   if ((window as any).cv) {
  //     cvRef.current = (window as any).cv;
  //     return;
  //   }

  //   await new Promise<void>((resolve, reject) => {
  //     const script = document.createElement("script");
  //     script.src = "/opencv/opencv.js";
  //     script.async = true;
  //     script.onload = () => {
  //       const cv = (window as any).cv;
  //       if (!cv) return reject(new Error("OpenCV โหลดไม่สำเร็จ"));
  //       cv["onRuntimeInitialized"] = () => {
  //         cvRef.current = cv;
  //         resolve();
  //       };
  //     };
  //     script.onerror = () => reject(new Error("โหลด opencv.js ไม่สำเร็จ"));
  //     document.body.appendChild(script);
  //   });
  // }
  async function loadOpenCV() {
  if (typeof window === "undefined") return;

  const win = window as { cv?: unknown };
  // ready แล้ว
  if ((win.cv as any)?.Mat) {
    cvRef.current = win.cv as unknown;
    return;
  }

  await new Promise<void>((resolve, reject) => {
    const script = document.createElement("script");
    script.src = "/opencv/opencv.js";
    script.async = true;

    script.onload = () => {
      const win = window as { cv?: unknown };
      const cv = (win.cv as any);
      if (!cv) return reject(new Error("OpenCV โหลดแล้วแต่ window.cv ไม่มีค่า"));

      const waitReady = () => {
        const win = window as { cv?: unknown };
        if ((win.cv as any)?.Mat) {
          cvRef.current = win.cv as unknown;
          resolve();
        } else {
          setTimeout(waitReady, 50);
        }
      };

      // บาง build มี callback บาง build พร้อมทันที
      if ("onRuntimeInitialized" in cv) {
        cv.onRuntimeInitialized = () => waitReady();
      } else {
        waitReady();
      }
    };

    script.onerror = () => reject(new Error("โหลด /opencv/opencv.js ไม่สำเร็จ"));
    document.body.appendChild(script);
  });
}


  // Load Haar cascade file into OpenCV FS
  async function loadCascade() {
    const cv = cvRef.current as any;
    if (!cv) throw new Error("cv ยังไม่พร้อม");

    const cascadeUrl = "/opencv/haarcascade_frontalface_default.xml";
    const res = await fetch(cascadeUrl);
    if (!res.ok) throw new Error("โหลด cascade ไม่สำเร็จ");
    const data = new Uint8Array(await res.arrayBuffer());

    // เขียนไฟลลง OpenCV virtual FS
    const cascadePath = "haarcascade_frontalface_default.xml";
    try {
      (cv as any).FS_unlink(cascadePath);
    } catch {}
    (cv as any).FS_createDataFile("/", cascadePath, data, true, false, false);

    const faceCascade = new (cv as any).CascadeClassifier();
    const loaded = faceCascade.load(cascadePath);
    if (!loaded) throw new Error("cascade load() ไม่สำเร็จ");
    faceCascadeRef.current = faceCascade;
  }

  // 3) Load ONNX model + classes
  async function loadModel() {
    // prefer a fixed model if the initializer-cleaned model exists
    let modelPath = "/models/onnx_model.fixed.onnx";
    try {
      const check = await fetch(modelPath, { method: "HEAD" });
      if (!check.ok) modelPath = "/models/onnx_model.onnx";
    } catch {
      modelPath = "/models/onnx_model.onnx";
    }

    const session = await ort.InferenceSession.create(modelPath, { executionProviders: ["wasm"] });
    sessionRef.current = session;

    // log input/output names to help debug shape/name mismatches
    console.info("ONNX session loaded. inputs=", session.inputNames, "outputs=", session.outputNames);

    const clsRes = await fetch("/models/classes.json");
    if (!clsRes.ok) throw new Error("โหลด classes.json ไม่สำเร็จ");
    classesRef.current = await clsRes.json();
  }

  // 4) Start camera
  async function startCamera() {
    setStatus("ขอสิทธิ์กล้อง...");
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user" },
      audio: false,
    });
    if (!videoRef.current) return;
    videoRef.current.srcObject = stream;
    streamRef.current = stream;
    await videoRef.current.play();
    setStatus("กำลังทำงาน...");
    setIsRecording(true);
    requestAnimationFrame(loop);
  }

  // 4.5) Stop camera
  function stopCamera() {
    loopRef.current = false;
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setStatus("ปิดกล้องแล้ว");
    setEmotion("-");
    setConf(0);
    setIsRecording(false);
  }

  // 5) Preprocess face ROI -> tensor
  // Convert to grayscale 48x48 to match model input shape [1,1,48,48]
  function preprocessToTensor(faceCanvas: HTMLCanvasElement) {
    const size = 48;

    // Try using OpenCV.js if available for equalization
    const cv = cvRef.current as any;
    if (cv && (cv as any).Mat) {
      const src = cv.imread(faceCanvas);
      const dst = new cv.Mat();
      const dsize = new cv.Size(size, size);
      cv.resize(src, dst, dsize, 0, 0, cv.INTER_LINEAR);

      const gray = new cv.Mat();
      cv.cvtColor(dst, gray, cv.COLOR_RGBA2GRAY);
      cv.equalizeHist(gray, gray);

      const data = new Uint8Array(gray.data);
      const float = new Float32Array(1 * 1 * size * size);
      for (let i = 0; i < size * size; i++) {
        float[i] = data[i] / 255;
      }

      src.delete();
      dst.delete();
      gray.delete();

      return new ort.Tensor("float32", float, [1, 1, size, size]);
    }

    // Fallback: canvas resize + grayscale
    const tmp = document.createElement("canvas");
    tmp.width = size;
    tmp.height = size;
    const ctx = tmp.getContext("2d")!;
    ctx.drawImage(faceCanvas, 0, 0, size, size);

    const imgData = ctx.getImageData(0, 0, size, size).data; // RGBA
    const float = new Float32Array(1 * 1 * size * size);
    for (let i = 0; i < size * size; i++) {
      const r = imgData[i * 4 + 0];
      const g = imgData[i * 4 + 1];
      const b = imgData[i * 4 + 2];
      const y = 0.299 * r + 0.587 * g + 0.114 * b;
      float[i] = y / 255;
    }

    return new ort.Tensor("float32", float, [1, 1, size, size]);
  }

  // 6) Softmax
  function softmax(logits: Float32Array) {
    let max = -Infinity;
    for (const v of logits) max = Math.max(max, v);
    const exps = logits.map((v) => Math.exp(v - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map((v) => v / sum);
  }

  // 7) Main loop
  async function loop() {
    try {
      // คำนวณ FPS - อัปเดตทุก 30 เฟรม
      frameCountRef.current++;
      if (frameCountRef.current % 30 === 0) {
        setFps(30); // แสดงประมาณ 30 FPS
      }

      const cv = cvRef.current as any;
      const faceCascade = faceCascadeRef.current as any;
      const session = sessionRef.current;
      const classes = classesRef.current;

      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (!cv || !faceCascade || !session || !classes || !video || !canvas) {
        if (loopRef.current) requestAnimationFrame(loop);
        return;
      }

      const ctx = canvas.getContext("2d")!;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      ctx.drawImage(video, 0, 0);

      // OpenCV: read frame
      const src = cv.imread(canvas);
      const gray = new cv.Mat();
      cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);

      const faces = new cv.RectVector();
      const msize = new cv.Size(50, 50);
      const maxsize = new cv.Size(Math.min(gray.rows, gray.cols), Math.min(gray.rows, gray.cols));
      faceCascade.detectMultiScale(gray, faces, 1.05, 5, 0, msize, maxsize);

      // วาดกรอบ + เลือกใบหน้าที่ใหญ่สุด
      let bestRect: any = null;
      let bestArea = 0;
      const MIN_FACE_SIZE = 100; // ขนาดใบหน้าต่ำสุด (px)
      const MAX_FACE_SIZE = Math.min(canvas.width, canvas.height) * 0.8; // ขนาดใบหน้าสูงสุด

      for (let i = 0; i < faces.size(); i++) {
        const r = faces.get(i);
        const area = r.width * r.height;
        // ตรวจสอบว่าใบหน้าอยู่ในช่วงขนาดที่เหมาะสม
        if (area >= MIN_FACE_SIZE * MIN_FACE_SIZE && area <= MAX_FACE_SIZE * MAX_FACE_SIZE) {
          if (area > bestArea) {
            bestArea = area;
            bestRect = r;
          }
        }
        ctx.strokeStyle = "lime";
        ctx.lineWidth = 2;
        ctx.strokeRect(r.x, r.y, r.width, r.height);
      }

      if (bestRect) {
        // ensure rect has valid size (integers) and clamp to canvas bounds
        const srcCanvasW = canvas.width;
        const srcCanvasH = canvas.height;
        let sx = Math.max(0, Math.floor(bestRect.x));
        let sy = Math.max(0, Math.floor(bestRect.y));
        let sWidth = Math.max(1, Math.floor(bestRect.width));
        let sHeight = Math.max(1, Math.floor(bestRect.height));

        // clamp width/height so they don't exceed canvas
        if (sx + sWidth > srcCanvasW) sWidth = Math.max(1, srcCanvasW - sx);
        if (sy + sHeight > srcCanvasH) sHeight = Math.max(1, srcCanvasH - sy);

        if (sWidth <= 0 || sHeight <= 0) {
          console.warn("Skipping frame due to invalid bestRect:", bestRect);
          src.delete();
          gray.delete();
          faces.delete();
          requestAnimationFrame(loop);
          return;
        }

        // crop face into a small canvas (destination uses clamped integer sizes)
        const faceCanvas = document.createElement("canvas");
        faceCanvas.width = sWidth;
        faceCanvas.height = sHeight;
        const fctx = faceCanvas.getContext("2d")!;
        fctx.drawImage(
          canvas,
          sx,
          sy,
          sWidth,
          sHeight,
          0,
          0,
          sWidth,
          sHeight
        );

        // run onnx
        let input: ort.Tensor | null = null;
        try {
          input = preprocessToTensor(faceCanvas);
        } catch (err) {
          // if preprocessing fails (e.g. zero-size), skip this frame
          console.warn("preprocessToTensor failed:", err);
          src.delete();
          gray.delete();
          faces.delete();
          requestAnimationFrame(loop);
          return;
        }

        if (!input) {
          src.delete();
          gray.delete();
          faces.delete();
          requestAnimationFrame(loop);
          return;
        }

        // ชื่อ input/output อาจต่างกันตามการ export
        // วิธีง่าย: ใช้ key ตัวแรกของ session.inputNames
        const feeds: Record<string, ort.Tensor> = {};
        feeds[session.inputNames[0]] = input;

        const out = await session.run(feeds);
        const outName = session.outputNames[0];
        const logits = out[outName].data as Float32Array;

        const probs = softmax(logits);
        let maxIdx = 0;
        for (let i = 1; i < probs.length; i++) {
          if (probs[i] > probs[maxIdx]) maxIdx = i;
        }

        // เพิ่ม confidence threshold (ต้องมีความมั่นใจ > 40%)
        const CONFIDENCE_THRESHOLD = 0.4;
        if (probs[maxIdx] >= CONFIDENCE_THRESHOLD) {
          setEmotion(classes[maxIdx] ?? `class_${maxIdx}`);
          setConf(probs[maxIdx] ?? 0);
        } else {
          setEmotion("Unclear");
          setConf(0);
        }

        ctx.fillStyle = "rgba(0,0,0,0.6)";
        ctx.fillRect(bestRect.x, Math.max(0, bestRect.y - 28), 220, 28);
        ctx.fillStyle = "white";
        ctx.font = "16px sans-serif";
        ctx.fillText(
          `${classes[maxIdx]} ${(probs[maxIdx] * 100).toFixed(1)}%`,
          bestRect.x + 6,
          bestRect.y - 8
        );
      }

      // cleanup
      src.delete();
      gray.delete();
      faces.delete();

      if (loopRef.current) requestAnimationFrame(loop);
    } catch (e: unknown) {
      const errMsg = e instanceof Error ? e.message : String(e);
      setStatus(`ผิดพลาด: ${errMsg}`);
    }
  }

  // Boot sequence + cleanup
  useEffect(() => {
    loopRef.current = true;
    (async () => {
      try {
        setStatus("กำลังโหลด OpenCV...");
        await loadOpenCV();

        setStatus("กำลังโหลด Haar cascade...");
        await loadCascade();

        setStatus("กำลังโหลดโมเดล ONNX...");
        await loadModel();

        setStatus("พร้อม เริ่มกดปุ่ม Start");
      } catch (e: unknown) {
        const errMsg = e instanceof Error ? e.message : String(e);
        setStatus(`เริ่มต้นไม่สำเร็จ: ${errMsg}`);
      }
    })();
    return () => {
      loopRef.current = false;
    };
  }, []);

  return (
    <main className="min-h-screen bg-linear-to-br from-slate-900 via-purple-900 to-slate-900 p-8">
      <div className="max-w-5xl mx-auto">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-5xl font-bold bg-linear-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent mb-2">
            Face Emotion Detection
          </h1>
          <p className="text-gray-300 text-lg">Real-time emotion recognition with OpenCV & YOLO11</p>
        </div>

        {/* Status Card */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          {/* Status */}
          <div className="bg-linear-to-br from-slate-800 to-slate-900 rounded-2xl p-6 border border-purple-500/30 backdrop-blur-sm">
            <div className="text-sm text-gray-400 mb-2">Status</div>
            <div className="text-xl font-semibold text-blue-300 break">{status}</div>
          </div>

          {/* Emotion Display */}
          <div className="bg-linear-to-br from-slate-800 to-slate-900 rounded-2xl p-6 border border-purple-500/30 backdrop-blur-sm">
            <div className="text-sm text-gray-400 mb-2">Detected Emotion</div>
            <div className="text-4xl font-bold bg-linear-to-r from-pink-400 to-rose-400 bg-clip-text text-transparent">
              {emotion}
            </div>
          </div>

          {/* Confidence */}
          <div className="bg-linear-to-br from-slate-800 to-slate-900 rounded-2xl p-6 border border-purple-500/30 backdrop-blur-sm">
            <div className="text-sm text-gray-400 mb-2">Confidence</div>
            <div className="flex items-end gap-2">
              <div className="text-3xl font-bold text-emerald-400">{(conf * 100).toFixed(1)}%</div>
            </div>
            <div className="mt-3 h-2 bg-slate-700 rounded-full overflow-hidden">
              <div
                className="h-full bg-linear-to-r from-emerald-400 to-teal-400 transition-all duration-500"
                style={{ width: `${conf * 100}%` }}
              />
            </div>
          </div>

          {/* FPS Display */}
          <div className="bg-linear-to-br from-slate-800 to-slate-900 rounded-2xl p-6 border border-cyan-500/30 backdrop-blur-sm">
            <div className="text-sm text-gray-400 mb-2">Frame Rate</div>
            <div className="text-3xl font-bold text-cyan-400">{fps}</div>
            <div className="text-xs text-gray-500 mt-1">FPS</div>
          </div>
        </div>

        {/* Control Section */}
        <div className="flex gap-4 mb-8 justify-center flex-wrap">
          <button
            className="px-8 py-3 rounded-xl font-semibold bg-linear-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 text-white transition-all duration-300 transform hover:scale-105 shadow-lg hover:shadow-2xl active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed"
            onClick={startCamera}
            disabled={isRecording}
          >
            ▶ Start Camera
          </button>
          {isRecording && (
            <button
              className="px-8 py-3 rounded-xl font-semibold bg-linear-to-r from-red-500 to-rose-600 hover:from-red-600 hover:to-rose-700 text-white transition-all duration-300 transform hover:scale-105 shadow-lg hover:shadow-2xl active:scale-95"
              onClick={stopCamera}
            >
              ⏹ Stop Camera
            </button>
          )}
        </div>

        {/* Canvas Display */}
        <div className="bg-linear-to-br from-slate-800 to-slate-900 rounded-3xl p-2 border-2 border-purple-500/40 backdrop-blur-sm shadow-2xl overflow-hidden">
          <video ref={videoRef} className="hidden" playsInline />
          <canvas
            ref={canvasRef}
            className="w-full rounded-2xl bg-black/50"
          />
        </div>

        {/* Info Section */}
        <div className="mt-8 text-center">
          <p className="text-gray-400 text-sm">
            💡 Click <span className="text-blue-400 font-semibold">Start Camera</span> to begin emotion detection
          </p>
        </div>
      </div>
    </main>
  );
}