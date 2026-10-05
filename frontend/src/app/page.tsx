"use client";

import React, { useState, useRef } from "react";
import DrawingCanvas from "@/components/DrawingCanvas";

export default function Home() {
  const [inputMode, setInputMode] = useState<"Draw" | "Upload">("Draw");
  const [blob, setBlob] = useState<Blob | null>(null);
  const [clearTrigger, setClearTrigger] = useState(0);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";

  const handleClear = () => {
    setBlob(null);
    setClearTrigger(c => c + 1);
    setResult(null);
    setError(null);
  };

  const handleUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setBlob(e.target.files[0]);
    }
  };

  const handleSolve = async () => {
    if (!blob) {
      setError("Please draw or upload an equation first.");
      return;
    }
    
    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append("file", blob, "equation.png");

    try {
      const res = await fetch(`${API_URL}/solve`, {
        method: "POST",
        body: formData,
      });
      
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error || "Failed to solve equation.");
      }
      
      setResult(data);
    } catch (err: any) {
      setError(err.message || "An unexpected error occurred.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen p-8 max-w-6xl mx-auto flex flex-col gap-8">
      <header className="flex justify-between items-end border-b border-card-border pb-4">
        <div>
          <h1 className="text-2xl font-bold tracking-widest">&Sigma; MATHVISION</h1>
          <p className="text-sm text-gray-400">HANDWRITTEN MATHEMATICS, UNDERSTOOD.</p>
        </div>
        <div className="flex gap-4 text-sm font-mono text-gray-400">
          <span className="hover:text-white cursor-pointer">About</span>
          <span className="flex items-center gap-2 hover:text-white cursor-pointer">
            Model <span className="w-2 h-2 rounded-full bg-accent inline-block shadow-[0_0_8px_#b4ff00]"></span> ONLINE
          </span>
        </div>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Left Side: Input Workspace */}
        <section className="mathvision-card flex flex-col gap-4">
          <div className="flex justify-between items-center mb-2">
            <h2 className="text-sm tracking-widest text-gray-300 font-mono">WRITE YOUR EQUATION</h2>
            <div className="flex border border-card-border rounded bg-black overflow-hidden text-xs font-mono">
              <button 
                className={`px-4 py-1 ${inputMode === "Draw" ? "bg-accent text-black" : "text-gray-400 hover:text-white"}`}
                onClick={() => { setInputMode("Draw"); handleClear(); }}
              >
                Draw
              </button>
              <button 
                className={`px-4 py-1 ${inputMode === "Upload" ? "bg-accent text-black" : "text-gray-400 hover:text-white"}`}
                onClick={() => { setInputMode("Upload"); handleClear(); }}
              >
                Upload
              </button>
            </div>
          </div>

          <div className="bg-white rounded p-1 mb-2">
            {inputMode === "Draw" ? (
              <DrawingCanvas 
                onDrawEnd={setBlob} 
                onClear={() => setBlob(null)} 
                clearTrigger={clearTrigger} 
              />
            ) : (
              <div className="h-[150px] flex flex-col justify-center items-center text-black border-2 border-dashed border-gray-300 rounded cursor-pointer"
                   onClick={() => fileInputRef.current?.click()}>
                {blob ? (
                  <p className="font-mono">{blob instanceof File ? blob.name : "Image selected"}</p>
                ) : (
                  <p className="font-mono">Click to upload image (PNG, JPG, WEBP)</p>
                )}
                <input type="file" className="hidden" ref={fileInputRef} accept="image/png, image/jpeg, image/webp" onChange={handleUpload} />
              </div>
            )}
          </div>

          <div className="flex gap-4">
            <button 
              className="flex-1 bg-card-border text-white py-2 rounded font-mono text-sm hover:bg-gray-700 transition"
              onClick={handleClear}
            >
              CLEAR
            </button>
            <button 
              className="flex-1 bg-accent text-black font-bold py-2 rounded font-mono text-sm hover:bg-[#c9ff33] transition disabled:opacity-50 disabled:cursor-not-allowed"
              disabled={!blob || loading}
              onClick={handleSolve}
            >
              {loading ? "SOLVING..." : "SOLVE EQUATION"}
            </button>
          </div>
          
          {error && <div className="text-red-400 text-sm font-mono mt-2 bg-red-900/20 p-2 rounded">{error}</div>}
        </section>

        {/* Right Side: Result */}
        <section className="mathvision-card flex flex-col gap-6">
          <h2 className="text-sm tracking-widest text-gray-300 font-mono mb-[-1rem]">MATHVISION RESULT</h2>
          
          {result ? (
            <>
              <div className="flex flex-col gap-1">
                <p className="text-gray-400 text-xs font-mono">RECOGNIZED EQUATION</p>
                <div className="text-2xl font-mono p-3 bg-black rounded border border-card-border">
                  {result.equation || <span className="text-gray-600">No equation</span>}
                </div>
              </div>
              
              <div className="flex flex-col gap-1">
                <p className="text-gray-400 text-xs font-mono">SOLUTION</p>
                <div className="text-2xl text-accent font-bold font-mono p-3 bg-black rounded border border-card-border">
                  {result.solution}
                </div>
              </div>

              <div className="flex gap-4">
                <div className="flex-1">
                  <p className="text-gray-400 text-xs font-mono mb-1">CONFIDENCE</p>
                  <div className="text-lg font-mono">
                    {(result.confidence * 100).toFixed(1)}%
                  </div>
                </div>
                <div className="flex-1">
                  <p className="text-gray-400 text-xs font-mono mb-1">INFERENCE TIME</p>
                  <div className="text-lg font-mono">
                    {result.inference_time_ms.toFixed(0)} ms
                  </div>
                </div>
              </div>

              {result.steps && result.steps.length > 0 && (
                <div className="flex flex-col gap-2 mt-2">
                  <p className="text-gray-400 text-xs font-mono border-b border-card-border pb-1">STEP-BY-STEP SOLUTION</p>
                  <div className="flex flex-col gap-2">
                    {result.steps.map((step: any, i: number) => (
                      <div key={i} className="text-sm font-mono bg-black p-2 rounded border border-card-border flex flex-col">
                        <span className="text-gray-400 text-xs mb-1">{step.operation}</span>
                        <div className="flex justify-between items-center">
                          <span className="text-gray-300">{step.before}</span>
                          <span className="text-accent">&rarr; {step.after}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {result.symbols && result.symbols.length > 0 && (
                <div className="flex flex-col gap-2 mt-2">
                  <p className="text-gray-400 text-xs font-mono border-b border-card-border pb-1">RECOGNITION INSPECTION</p>
                  <div className="flex flex-wrap gap-2">
                    {result.symbols.map((sym: any, i: number) => (
                      <div key={i} className="bg-black border border-card-border p-2 rounded flex flex-col items-center min-w-[50px]">
                        <span className="text-xl font-mono">{sym.symbol}</span>
                        <div className="w-full h-1 bg-card-border mt-1 relative rounded overflow-hidden">
                          <div className="absolute top-0 left-0 h-full bg-accent" style={{ width: `${sym.confidence * 100}%` }}></div>
                        </div>
                        <span className="text-[10px] text-gray-500 mt-1">{(sym.confidence * 100).toFixed(0)}%</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </>
          ) : (
            <div className="flex-1 flex items-center justify-center text-gray-500 font-mono text-sm bg-black rounded border border-card-border min-h-[300px]">
              AWAITING INPUT...
            </div>
          )}
        </section>
      </div>

      <div className="mathvision-card">
        <h2 className="text-sm tracking-widest text-gray-300 font-mono mb-4">PIPELINE</h2>
        <div className="flex flex-wrap justify-between items-center font-mono text-xs text-gray-400">
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-full bg-white"></span> INPUT
          </div>
          <span className="text-card-border">&rarr;</span>
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-full bg-gray-500"></span> PREPROCESS
          </div>
          <span className="text-card-border">&rarr;</span>
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-full bg-gray-500"></span> SEGMENT
          </div>
          <span className="text-card-border">&rarr;</span>
          <div className="flex items-center gap-2 text-accent">
            <span className="w-3 h-3 rounded-full bg-accent shadow-[0_0_5px_#b4ff00]"></span> CNN RECOGNITION
          </div>
          <span className="text-card-border">&rarr;</span>
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-full bg-gray-500"></span> RECONSTRUCT
          </div>
          <span className="text-card-border">&rarr;</span>
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-full bg-gray-500"></span> SOLVE
          </div>
        </div>
      </div>
    </main>
  );
}
