"use client";

import React, { useEffect, useRef, useState } from "react";

export default function CameraPage() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const hiddenCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const overlayRef = useRef<HTMLCanvasElement | null>(null);
  const [status, setStatus] = useState("loading resources...");
  const [label, setLabel] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
    let session: any = null;
    let classes: string[] = [];
    let classifier: any = null;

    async function init() {
      try {
        setStatus("loading ONNX runtime...");
        const ort = await import("onnxruntime-web");
        setStatus("loading model...");
        session = await ort.InferenceSession.create("/models/emotion_yolo11n_cls.onnx");

        const resp = await fetch("/models/classes.json");
        classes = await resp.json();

        setStatus("loading OpenCV...");
        await loadOpenCv();

        // load cascade xml into FS and create classifier
        const xml = await fetch("/opencv/haarcascade_frontalface_default.xml").then((r) => r.arrayBuffer());
        const bytes = new Uint8Array(xml);
        // @ts-ignore
        cv.FS_createDataFile("/", "haarcascade_frontalface_default.xml", bytes, true, false, false);
        // @ts-ignore
        classifier = new cv.CascadeClassifier();
        // @ts-ignore
        classifier.load("haarcascade_frontalface_default.xml");

        setStatus("starting camera...");
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (!mounted) return;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
        }

        setStatus("running detection");
        runLoop();

        async function runLoop() {
          if (!mounted) return;
          try {
            if (videoRef.current && hiddenCanvasRef.current && overlayRef.current) {
              const v = videoRef.current;
              const hc = hiddenCanvasRef.current;
              const oc = overlayRef.current;
              const vw = v.videoWidth || 640;
              const vh = v.videoHeight || 480;
              hc.width = vw;
              hc.height = vh;
              oc.width = vw;
              oc.height = vh;
              const hctx = hc.getContext("2d")!;
              const octx = oc.getContext("2d")!;
              hctx.drawImage(v, 0, 0, vw, vh);

              // use OpenCV to detect faces
              // @ts-ignore
              const src = cv.imread(hc);
              // @ts-ignore
              const gray = new cv.Mat();
              // @ts-ignore
              cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
              // @ts-ignore
              const faces = new cv.RectVector();
              // @ts-ignore
              const size = new cv.Size(0, 0);
              // @ts-ignore
              classifier.detectMultiScale(gray, faces, 1.1, 3, 0, size, size);

              octx.clearRect(0, 0, vw, vh);

              for (let i = 0; i < faces.size(); i++) {
                const r = faces.get(i);
                octx.strokeStyle = "#00FF00";
                octx.lineWidth = 2;
                octx.strokeRect(r.x, r.y, r.width, r.height);

                // crop face to temporary canvas
                const cropCanvas = document.createElement("canvas");
                cropCanvas.width = r.width;
                cropCanvas.height = r.height;
                const cctx = cropCanvas.getContext("2d")!;
                cctx.drawImage(hc, r.x, r.y, r.width, r.height, 0, 0, r.width, r.height);

                // resize to model input (224x224)
                const modelW = 224;
                const modelH = 224;
                const resized = document.createElement("canvas");
                resized.width = modelW;
                resized.height = modelH;
                const rctx = resized.getContext("2d")!;
                rctx.drawImage(cropCanvas, 0, 0, modelW, modelH);

                const imgData = rctx.getImageData(0, 0, modelW, modelH);
                // convert to float32 CHW normalized [0,1]
                const floatData = new Float32Array(3 * modelH * modelW);
                // CHW
                let ptr = 0;
                for (let c = 0; c < 3; c++) {
                  for (let y = 0; y < modelH; y++) {
                    for (let x = 0; x < modelW; x++) {
                      const idx = (y * modelW + x) * 4;
                      const value = imgData.data[idx + (2 - c)];
                      floatData[ptr++] = value / 255.0;
                    }
                  }
                }

                // run ONNX model
                try {
                  const ort = await import("onnxruntime-web");
                  const inputName = session.inputNames ? session.inputNames[0] : Object.keys(session.inputMetadata)[0];
                  const tensor = new ort.Tensor("float32", floatData, [1, 3, modelH, modelW]);
                  const feeds: any = {};
                  feeds[inputName] = tensor;
                  const output = await session.run(feeds);
                  const outArr = Object.values(output)[0].data as Float32Array | number[];
                  let maxI = 0;
                  let maxV = -Infinity;
                  for (let k = 0; k < outArr.length; k++) {
                    if (outArr[k] > maxV) {
                      maxV = outArr[k] as number;
                      maxI = k;
                    }
                  }
                  const lbl = classes[maxI] ?? String(maxI);
                  if (mounted) setLabel(lbl);
                } catch (e) {
                  console.error("model run error", e);
                }
              }

              src.delete();
              gray.delete();
              faces.delete();
            }
          } catch (err) {
            console.error(err);
          }
          requestAnimationFrame(runLoop);
        }
      } catch (err) {
        console.error(err);
        setStatus("error: " + (err as any).message ?? String(err));
      }
    }

    init();

    return () => {
      mounted = false;
    };
  }, []);

  return (
    <div style={{ padding: 12 }}>
      <h2>Camera Emotion Detection</h2>
      <p>Status: {status}</p>
      <p>Detected: {label ?? "-"}</p>
      <div style={{ position: "relative", width: 640, height: 480 }}>
        <video ref={videoRef} style={{ width: "100%", height: "100%" }} playsInline muted />
        <canvas ref={overlayRef} style={{ position: "absolute", left: 0, top: 0 }} />
      </div>
      <canvas ref={hiddenCanvasRef} style={{ display: "none" }} />
    </div>
  );
}

async function loadOpenCv(): Promise<void> {
  if (typeof window === "undefined") return Promise.resolve();
  // @ts-ignore
  if ((window as any).cv && (window as any).cv['onRuntimeInitialized'] && (window as any).cv['onRuntimeInitialized_called']) {
    return Promise.resolve();
  }

  return new Promise((resolve, reject) => {
    // if already loaded on window, wait for runtime init
    // @ts-ignore
    if ((window as any).cv) {
      // @ts-ignore
      const wcv = (window as any).cv;
      if (wcv['onRuntimeInitialized']) {
        wcv['onRuntimeInitialized'] = () => resolve();
        return;
      }
      // if no onRuntimeInitialized, assume ready
      resolve();
      return;
    }

    const s = document.createElement("script");
    s.src = "/opencv/opencv.js";
    s.async = true;
    s.onload = () => {
      // wait until cv is defined and runtime initialized
      const checkCv = () => {
        // @ts-ignore
        const wcv = (window as any).cv;
        if (!wcv) {
          setTimeout(checkCv, 50);
          return;
        }
        if (wcv['onRuntimeInitialized']) {
          wcv['onRuntimeInitialized'] = () => resolve();
        } else {
          resolve();
        }
      };
      checkCv();
    };
    s.onerror = (e) => reject(e);
    document.body.appendChild(s);
  });
}
