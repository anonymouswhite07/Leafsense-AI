import React, { useState, useRef, useEffect } from 'react';
import { UploadCloud, Leaf, AlertCircle, CheckCircle, Loader2, Globe, Clock, X, Camera, Aperture, MessageSquare, Send, Bot, FileText, Zap, ThumbsUp, ThumbsDown, Info, LayoutDashboard, MapPin, AlertTriangle, Download } from 'lucide-react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';

interface TTAImageResult {
  disease: string;
  confidence: number;
  solution: string;
  cam_base64?: string;
  is_uncertain: boolean;
  top3_predictions: {disease: string, confidence: number}[];
  consistency_score: number;
  final_decision: "stable" | "unstable";
}

interface Prediction {
  id: string;
  images: string[];
  final_prediction: string;
  confidence: number;
  solution: string;
  cross_image_consistency: number;
  status: "stable" | "conflict";
  image_results: TTAImageResult[];
  timestamp: string;
}

interface ChatMessage {
  id: string;
  sender: 'user' | 'bot';
  text: string;
}

interface ModelInfo {
  version: string;
  accuracy: string;
  last_trained: string;
}

interface AnalyticsData {
  total_scans: number;
  distribution: {name: string, value: number}[];
  regional: {region: string, scans: number}[];
  trends: {date: string, scans: number}[];
  global_alerts: string[];
}

const COLORS = ['#10b981', '#f59e0b', '#ef4444', '#3b82f6', '#8b5cf6', '#64748b'];

const translations = {
  en: {
    title: "LeafSense AI",
    heading: "Identify Plant Diseases in Seconds",
    subheading: "Upload an image of a leaf to instantly detect potential diseases, get a confidence score, and learn actionable next steps to save your crops.",
    uploadText: "Click or drag and drop to upload",
    uploadFormat: "PNG, JPG or JPEG up to 10MB",
    openCamera: "Scan with Camera",
    captureBtn: "Capture Frame",
    closeCamera: "Close Camera",
    analyzeBtn: "Analyze Leaf",
    cancelBtn: "Cancel",
    runningModel: "Running neural networks...",
    failTitle: "Analysis Failed",
    tryAgain: "Try again",
    validated: "Validated",
    remedy: "Suggested Remedy",
    scanAnother: "Scan Another Leaf",
    history: "Recent Scans",
    clearHistory: "Clear All",
    noHistory: "No recent scans available.",
    cameraError: "Unable to access camera.",
    chatHelpText: "Ask AI Agronomist...",
    askBotPlaceholder: "E.g., What pesticide should I use?",
    chatTitle: "AI Agronomist Chat",
    downloadReport: "Download PDF",
    downloadingReport: "Generating...",
    heatmapTitle: "AI Vision Heatmap",
    feedbackCorrect: "Correct",
    feedbackWrong: "Wrong",
    feedbackSubmit: "Submit Correction",
    feedbackThanks: "Thanks! Data captured for retraining.",
    uncertainWarning: "Prediction uncertain. Please ensure the leaf is clearly visible and well-lit.",
    selectCorrectLabel: "What is the actual disease?",
    dashboard: "Intelligence Dashboard",
    exportData: "Export Anonymized Data",
    activeOutbreaks: "Active Outbreak Alerts",
    diseaseDistribution: "Disease Distribution",
    recentTrends: "Scan Activity (7 Days)",
    gettingLocation: "Acquiring GPS...",
    locationGranted: "Geo-Enabled"
  },
  ta: {
    title: "இலைஉணர்வு AI",
    heading: "தாவர நோய்கள் கண்டறிதல்",
    subheading: "சாத்தியமான நோய்களை உடனடியாகக் கண்டறிய இலை படத்தைப் பதிவேற்றி காப்பாற்றவும்.",
    uploadText: "பதிவேற்ற சொடுக்கவும் / இழுத்து விடவும்",
    uploadFormat: "10MB வரை PNG, JPG",
    openCamera: "காமிராவை திறக்க",
    captureBtn: "படம் பிடி",
    closeCamera: "மூடு",
    analyzeBtn: "இலையை பகுப்பாய்வு செய்",
    cancelBtn: "ரத்து செய்",
    runningModel: "மதிப்பீடு நடக்கிறது...",
    failTitle: "பகுப்பாய்வு தோல்வியடைந்தது",
    tryAgain: "மீண்டும் முயற்சி செய்",
    validated: "உறுதிசெய்யப்பட்டது",
    remedy: "தீர்வு",
    scanAnother: "மற்றொன்றை ஸ்கேன் செய்",
    history: "சமீபத்திய ஸ்கேன்கள்",
    clearHistory: "அழி",
    noHistory: "ஸ்கேன்கள் இல்லை.",
    cameraError: "காமிராவை அணுக முடியவில்லை.",
    chatHelpText: "AI நிபுணரிடம் கேளுங்கள்...",
    askBotPlaceholder: "எந்த பூச்சிக்கொல்லியைப் பயன்படுத்த வேண்டும்?",
    chatTitle: "AI அரட்டை",
    downloadReport: "பதிவிறக்கு",
    downloadingReport: "உருவாக்குகிறது...",
    heatmapTitle: "AI பார்வை பகுப்பாய்வு",
    feedbackCorrect: "சரி",
    feedbackWrong: "தவறு",
    feedbackSubmit: "சமர்ப்பிக்கவும்",
    feedbackThanks: "நன்றி! தரவு சேமிக்கப்பட்டது.",
    uncertainWarning: "கணிப்பு நிச்சயமற்றது. இலை தெளிவாகவும் வெளிச்சமாகவும் இருப்பதை உறுதி செய்யவும்.",
    selectCorrectLabel: "உண்மையான நோய் என்ன?",
    dashboard: "நுண்ணறிவு டாஷ்போர்டு",
    exportData: "தரவு ஏற்றுமதி",
    activeOutbreaks: "நோய் வெடிப்பு எச்சரிக்கைகள்",
    diseaseDistribution: "நோய் பரவல்",
    recentTrends: "ஸ்கேன் செயல்பாடு (7 நாட்கள்)",
    gettingLocation: "ஜிபிஎஸ் பெறுகிறது...",
    locationGranted: "ஜிபிஎஸ் இயக்கப்பட்ட"
  }
};

const AVAILABLE_LABELS = [
  "Apple___Apple_scab", "Apple___healthy", "Banana___Black_Sigatoka",
  "Banana___healthy", "Corn___Common_rust", "Corn___healthy", 
  "Tomato___Bacterial_spot", "Tomato___healthy"
];

const API_BASE = import.meta.env.VITE_API_URL || (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1" ? "http://127.0.0.1:8000" : "/api");

function App() {
  const [files, setFiles] = useState<File[]>([]);
  const [previews, setPreviews] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<Prediction[]>([]);
  const [lang, setLang] = useState<'en' | 'ta'>('en');
  const [showHistory, setShowHistory] = useState(false);
  const [isCameraOpen, setIsCameraOpen] = useState(false);
  const [isDownloading, setIsDownloading] = useState(false);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  
  // Phase 15 States
  const [isDashboard, setIsDashboard] = useState(false);
  const [localAlerts, setLocalAlerts] = useState<string[]>([]);
  const [globalAlerts, setGlobalAlerts] = useState<string[]>([]);
  const [analytics, setAnalytics] = useState<AnalyticsData | null>(null);
  const [locationState, setLocationState] = useState<'idle' | 'fetching' | 'granted' | 'denied'>('idle');
  const [coords, setCoords] = useState<{lat: number, lng: number} | null>(null);
  
  const [feedbackState, setFeedbackState] = useState<'none' | 'wrong' | 'submitted'>('none');
  const [selectedCorrectLabel, setSelectedCorrectLabel] = useState(AVAILABLE_LABELS[0]);
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState("");
  const [isChatLoading, setIsChatLoading] = useState(false);
  const chatBottomRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const t = translations[lang];

  useEffect(() => {
    const savedHistory = localStorage.getItem('leafSenseHistory');
    if (savedHistory) setHistory(JSON.parse(savedHistory));
    
    fetch(`${API_BASE}/model-info`)
      .then(res => res.json())
      .then(data => setModelInfo(data))
      .catch(console.error);
      
    // Ask for Phase 15 Geolocation implicitly to track outbreaks on mount
    if ('geolocation' in navigator) {
      setLocationState('fetching');
      navigator.geolocation.getCurrentPosition(
         (pos) => { setCoords({ lat: pos.coords.latitude, lng: pos.coords.longitude }); setLocationState('granted'); },
         (err) => { setLocationState('denied'); console.error("Geo error:", err); }
      );
    }
      
    return () => stopCamera();
  }, []);

  useEffect(() => {
    if (isDashboard) {
      fetch(`${API_BASE}/analytics`)
        .then(res => res.json())
        .then(data => {
            setAnalytics(data);
            setGlobalAlerts(data.global_alerts || []);
        });
    }
  }, [isDashboard]);

  useEffect(() => {
    if (chatBottomRef.current) chatBottomRef.current.scrollIntoView({ behavior: "smooth" });
  }, [chatMessages]);

  const handleLanguageToggle = () => setLang(prev => (prev === 'en' ? 'ta' : 'en'));

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) setupFiles(Array.from(e.target.files).slice(0, 5));
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (e.dataTransfer.files) setupFiles(Array.from(e.dataTransfer.files).slice(0, 5));
  };

  const setupFiles = (selectedFiles: File[]) => {
    setFiles(prev => [...prev, ...selectedFiles].slice(0, 5)); 
    setPrediction(null); setError(null); setChatMessages([]); setFeedbackState('none'); setLocalAlerts([]);
    const newPreviews = selectedFiles.map(f => URL.createObjectURL(f));
    setPreviews(prev => [...prev, ...newPreviews].slice(0, 5));
    stopCamera();
  };

  const startCamera = async () => {
    setIsCameraOpen(true); setPrediction(null); setError(null); setChatMessages([]);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
      if (videoRef.current) videoRef.current.srcObject = stream;
    } catch (err) {
      console.error(t.cameraError); setIsCameraOpen(false);
    }
  };

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
    setIsCameraOpen(false);
  };

  const captureFrame = () => {
    if (videoRef.current && canvasRef.current) {
      const video = videoRef.current; const canvas = canvasRef.current;
      canvas.width = video.videoWidth; canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        canvas.toBlob((blob) => {
          if (blob) {
            const capturedFile = new File([blob], `camera_capture_${files.length}.jpg`, { type: "image/jpeg" });
            setupFiles([capturedFile]);
          }
        }, 'image/jpeg', 0.95);
      }
    }
  };

  const uploadImage = async () => {
    if (files.length === 0 || previews.length === 0) return;
    setLoading(true); setPrediction(null); setError(null); setChatMessages([]); setFeedbackState('none');
    const formData = new FormData(); 
    files.forEach(f => formData.append("files", f));
    if(coords) { formData.append("lat", coords.lat.toString()); formData.append("lng", coords.lng.toString()); }
    
    try {
      const response = await fetch(`${API_BASE}/predict`, { method: "POST", body: formData });
      if (!response.ok) throw new Error("Server error.");
      const data = await response.json();
      if (data.error) throw new Error(data.error);
      
      const newPrediction: Prediction = { 
        id: Date.now().toString(), images: previews,
        final_prediction: data.final_prediction, confidence: data.confidence,
        solution: data.solution, cross_image_consistency: data.cross_image_consistency,
        status: data.status, image_results: data.image_results,
        timestamp: new Date().toLocaleString()
      };
      
      setPrediction(newPrediction);
      if(data.alerts && data.alerts.length > 0) setLocalAlerts(data.alerts);
      
      if(data.status === "stable") {
        const newHistory = [newPrediction, ...history].slice(0, 5); 
        setHistory(newHistory); localStorage.setItem('leafSenseHistory', JSON.stringify(newHistory));
        setChatMessages([{ id: "w", sender: "bot", text: lang === 'en' ? `I noticed your crop might have ${data.final_prediction.replace(/___/g, ' - ').replace(/_/g, ' ')}. Ask me anything about how to treat it!` : `இதை எப்படி சிகிச்சை செய்வது என்று என்னைக் கேளுங்கள்!`}]);
      }
    } catch (err: any) {
      setError(err.message || "An error occurred.");
    } finally { setLoading(false); }
  };

  const submitFeedback = async (isCorrect: boolean) => {
    if(files.length === 0 || !prediction) return;
    try {
        const formData = new FormData();
        formData.append("file", files[0]);
        formData.append("predicted", prediction.final_prediction);
        formData.append("actual", isCorrect ? prediction.final_prediction : selectedCorrectLabel);
        formData.append("confidence", prediction.confidence.toString());
        await fetch(`${API_BASE}/feedback`, { method: "POST", body: formData });
        setFeedbackState('submitted');
    } catch(e) { console.error(e); }
  };

  const handleDownloadReport = async () => {
    if (files.length === 0 || !prediction) return;
    setIsDownloading(true);
    try {
      const formData = new FormData();
      formData.append("file", files[0]);
      formData.append("disease", prediction.final_prediction);
      formData.append("confidence", prediction.confidence.toString());
      formData.append("solution", prediction.solution);
      const response = await fetch(`${API_BASE}/report`, { method: "POST", body: formData });
      if (!response.ok) throw new Error("Failed to generate report.");
      const blob = await response.blob();
      const downloadUrl = window.URL.createObjectURL(blob);
      const tempLink = document.createElement("a");
      tempLink.href = downloadUrl;
      tempLink.setAttribute("download", `Report-${prediction.final_prediction}.pdf`);
      document.body.appendChild(tempLink);
      tempLink.click();
      window.URL.revokeObjectURL(downloadUrl);
      tempLink.remove();
    } catch (err: any) { alert("Error downloading report."); } finally { setIsDownloading(false); }
  };
  
  const handleExportData = async () => {
     try {
         const response = await fetch(`${API_BASE}/export`);
         const data = await response.json();
         const blob = new Blob([JSON.stringify(data, null, 2)], {type : 'application/json'});
         const url = window.URL.createObjectURL(blob);
         const a = document.createElement('a');
         a.href = url;
         a.download = `leafsense_export_${new Date().getTime()}.json`;
         a.click();
     } catch (e) { console.error("Export failed", e); }
  }

  const sendChatMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!chatInput.trim() || !prediction) return;
    const userMessage = chatInput.trim(); setChatInput("");
    setChatMessages(prev => [...prev, { id: Date.now().toString(), sender: 'user', text: userMessage }]);
    setIsChatLoading(true);
    try {
      const response = await fetch(`${API_BASE}/chat`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ message: userMessage, disease: prediction.final_prediction, language: lang }) });
      if (!response.ok) throw new Error("Chat server unreachable.");
      const data = await response.json();
      setChatMessages(prev => [...prev, { id: Date.now().toString(), sender: 'bot', text: data.reply }]);
    } catch (err) {
      setChatMessages(prev => [...prev, { id: Date.now().toString(), sender: 'bot', text: "Error connecting to AI." }]);
    } finally { setIsChatLoading(false); }
  };

  const reset = () => { setFiles([]); setPreviews([]); setPrediction(null); setError(null); setChatMessages([]); setFeedbackState('none'); setLocalAlerts([]); stopCamera(); };
  const clearHistory = () => { setHistory([]); localStorage.removeItem('leafSenseHistory'); };

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col font-sans">
      <nav className="w-full bg-white shadow-sm border-b border-gray-100 flex items-center justify-between px-8 py-4 sticky top-0 z-20">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 text-emerald-600 hover:opacity-80 cursor-pointer" onClick={() => setIsDashboard(false)}>
            <Leaf className="w-8 h-8" />
            <h1 className="text-xl font-bold tracking-tight">{t.title}</h1>
          </div>
          {modelInfo && !isDashboard && (
            <div className="hidden md:flex items-center gap-2 text-xs font-semibold bg-gray-100 px-3 py-1 rounded-full text-gray-500">
               <Info className="w-3 h-3" /> Model {modelInfo.version}
            </div>
          )}
          {locationState === 'fetching' && <div className="hidden lg:flex items-center gap-1.5 text-xs text-amber-600 font-semibold"><Loader2 className="w-3 h-3 animate-spin"/> {t.gettingLocation}</div>}
          {locationState === 'granted' && <div className="hidden lg:flex items-center gap-1.5 text-xs text-emerald-600 font-semibold"><MapPin className="w-3 h-3"/> {t.locationGranted}</div>}
        </div>
        <div className="flex items-center gap-4">
          <button onClick={() => setIsDashboard(!isDashboard)} className={`flex items-center gap-2 text-sm font-semibold transition-colors px-4 py-2 rounded-lg ${isDashboard ? 'bg-slate-900 text-white' : 'text-slate-600 bg-slate-100 hover:bg-slate-200'}`}>
             <LayoutDashboard className="w-4 h-4" /> <span className="hidden sm:inline">{t.dashboard}</span>
          </button>
          {!isDashboard && (
            <button onClick={() => setShowHistory(!showHistory)} className="flex items-center gap-1 sm:gap-2 text-sm font-medium text-gray-600 hover:text-emerald-600 transition-colors">
                <Clock className="w-4 h-4" /> <span className="hidden sm:inline">{t.history}</span>
            </button>
          )}
          <button onClick={handleLanguageToggle} className="flex items-center gap-2 text-sm font-medium bg-emerald-50 text-emerald-700 px-3 py-1.5 rounded-full hover:bg-emerald-100 transition-colors">
            <Globe className="w-4 h-4" /> {lang === 'en' ? 'தமிழ்' : 'English'}
          </button>
        </div>
      </nav>
      
      {/* Phase 15 Global Outbreak Alert Banner */}
      {globalAlerts.length > 0 && isDashboard && (
          <div className="w-full bg-red-600 text-white text-sm font-semibold sticky top-[73px] z-10 shadow border-b border-red-700">
             <div className="max-w-6xl mx-auto px-6 py-3 flex gap-4 overflow-x-auto whitespace-nowrap hide-scrollbar">
                {globalAlerts.map((a, i) => <div key={i} className="flex items-center gap-2 shrink-0"><AlertTriangle className="w-4 h-4"/> {a}</div>)}
             </div>
          </div>
      )}

      {showHistory && !isDashboard && (
        <div className="fixed inset-y-0 right-0 w-80 bg-white shadow-2xl z-30 border-l border-gray-100 p-6 flex flex-col animate-in slide-in-from-right-10 overflow-y-auto">
          {/* Omitted History Sidebar for brevity but fully functional. */}
          <div className="flex items-center justify-between mb-6">
            <h3 className="font-bold text-gray-900 text-lg">{t.history}</h3>
            <button onClick={() => setShowHistory(false)} className="text-gray-400 hover:text-gray-600"><X className="w-5 h-5" /></button>
          </div>
          {history.length === 0 ? <p className="text-gray-500 text-center text-sm">{t.noHistory}</p> : (
            <div className="flex flex-col gap-4 flex-grow">
              {history.map((item) => (
                <div key={item.id} className="bg-emerald-50/50 border border-emerald-100 rounded-xl p-3 flex gap-3">
                  <img src={item.images[0]} alt="scan" className="w-16 h-16 object-cover rounded-lg border border-emerald-200" />
                  <div className="flex-1 min-w-0">
                    <p className="font-semibold text-emerald-900 text-sm truncate" title={item.final_prediction}>{item.final_prediction.split('___')[1] || item.final_prediction}</p>
                    <p className="text-xs font-bold text-emerald-600 mt-1">{item.confidence}%</p>
                  </div>
                </div>
              ))}
              <button onClick={clearHistory} className="mt-auto text-sm text-red-500 font-medium hover:text-red-600">{t.clearHistory}</button>
            </div>
          )}
        </div>
      )}

      {/* INTELLIGENCE DASHBOARD VEW */}
      {isDashboard && analytics && (
          <main className="flex-grow bg-slate-50 p-8 w-full max-w-6xl mx-auto animate-in fade-in">
             <div className="flex justify-between items-center mb-8">
                 <div>
                    <h2 className="text-3xl font-bold text-slate-800">{t.dashboard}</h2>
                    <p className="text-slate-500 mt-1">Live tracking of {analytics.total_scans} regional crop disease scans.</p>
                 </div>
                 <button onClick={handleExportData} className="flex items-center gap-2 bg-emerald-600 text-white px-5 py-2.5 rounded-lg font-semibold shadow-sm hover:bg-emerald-700">
                    <Download className="w-4 h-4"/> {t.exportData}
                 </button>
             </div>
             
             <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
                {/* Outbreak Panel */}
                <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm col-span-1 md:col-span-2 lg:col-span-1">
                    <h3 className="font-bold text-slate-800 mb-4 flex items-center gap-2"><AlertTriangle className="text-orange-500 w-5 h-5"/> {t.activeOutbreaks}</h3>
                    {analytics.global_alerts.length === 0 ? (
                        <div className="h-40 flex items-center justify-center text-slate-400 bg-slate-50 rounded-xl border border-dashed border-slate-200">No active outbreaks detected.</div>
                    ) : (
                        <div className="flex flex-col gap-3">
                            {analytics.global_alerts.map((a, i) => (
                                <div key={i} className="p-4 bg-orange-50 border border-orange-200 text-orange-900 rounded-xl font-medium text-sm">{a}</div>
                            ))}
                        </div>
                    )}
                </div>

                {/* Pie Chart Distribution */}
                <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm col-span-1 md:col-span-2 lg:col-span-1">
                    <h3 className="font-bold text-slate-800 mb-4">{t.diseaseDistribution}</h3>
                    <div className="h-64 relative">
                        <ResponsiveContainer width="100%" height="100%">
                            <PieChart>
                                <Pie data={analytics.distribution} innerRadius={60} outerRadius={80} paddingAngle={5} dataKey="value">
                                    {analytics.distribution.map((_entry, index) => (
                                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                    ))}
                                </Pie>
                                <RechartsTooltip />
                            </PieChart>
                        </ResponsiveContainer>
                    </div>
                </div>
             </div>

             <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm w-full">
                 <h3 className="font-bold text-slate-800 mb-6">{t.recentTrends}</h3>
                 <div className="h-72">
                    <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={analytics.trends} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                            <defs>
                              <linearGradient id="colorScans" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#10b981" stopOpacity={0.3}/>
                                <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
                              </linearGradient>
                            </defs>
                            <XAxis dataKey="date" tick={{fontSize: 12, fill: '#64748b'}} tickLine={false} axisLine={false} />
                            <YAxis tick={{fontSize: 12, fill: '#64748b'}} tickLine={false} axisLine={false} />
                            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
                            <RechartsTooltip />
                            <Area type="monotone" dataKey="scans" stroke="#10b981" strokeWidth={3} fillOpacity={1} fill="url(#colorScans)" />
                        </AreaChart>
                    </ResponsiveContainer>
                 </div>
             </div>
          </main>
      )}

      {/* STANDARD SCANNER VIEW */}
      {!isDashboard && (
          <main className="flex-grow flex flex-col items-center justify-center p-6 w-full max-w-5xl mx-auto">
            {!prediction && (
              <div className="text-center mb-10">
                <h2 className="text-4xl font-extrabold text-gray-900 mb-4 tracking-tight">{t.heading}</h2>
                <p className="text-lg text-gray-600 max-w-2xl mx-auto leading-relaxed">{t.subheading}</p>
              </div>
            )}
            
            {localAlerts.length > 0 && !isDashboard && (
                <div className="w-full max-w-2xl mb-6 flex flex-col gap-3 animate-in fade-in slide-in-from-top-4">
                    {localAlerts.map((a,i) => (
                        <div key={i} className="bg-red-50 text-red-700 border border-red-200 p-4 rounded-xl flex items-center gap-3 shadow-sm">
                            <AlertTriangle className="w-6 h-6 shrink-0"/> <span className="font-semibold text-sm leading-tight">{a}</span>
                        </div>
                    ))}
                </div>
            )}

            <div className="w-full max-w-2xl bg-white p-8 rounded-3xl shadow-xl shadow-gray-200/50 border border-gray-100">
              <p className="text-center font-semibold text-emerald-600 mb-6 flex items-center justify-center gap-2">
                 <Camera className="w-4 h-4"/> Capture leaf from different angles for optimal accuracy (Max 5)
              </p>
              
              {!isCameraOpen && previews.length < 5 && !prediction && (
                <div className="flex flex-col sm:flex-row gap-6 mb-6">
                  <div 
                    onDragOver={(e) => e.preventDefault()} onDrop={handleDrop} onClick={() => fileInputRef.current?.click()}
                    className="flex-1 border-2 border-dashed border-emerald-200 rounded-2xl p-6 flex flex-col items-center justify-center bg-emerald-50/50 hover:bg-emerald-50 cursor-pointer transition-all duration-300 group"
                  >
                    <UploadCloud className="w-8 h-8 text-emerald-500 mb-2 group-hover:scale-110 transition-transform duration-300" />
                    <p className="text-emerald-900 font-semibold text-center text-sm">{t.uploadText}</p>
                    <input type="file" ref={fileInputRef} onChange={handleFileChange} className="hidden" accept="image/png, image/jpeg, image/jpg" multiple />
                  </div>
                  <div onClick={startCamera} className="flex-1 border-2 border-transparent rounded-2xl p-6 flex flex-col items-center justify-center bg-slate-900 hover:bg-slate-800 text-white cursor-pointer shadow-lg transform hover:-translate-y-1 transition-all group">
                    <Camera className="w-8 h-8 text-emerald-400 mb-2 group-hover:scale-110" />
                    <p className="font-semibold text-sm">{t.openCamera}</p>
                  </div>
                </div>
              )}

              {isCameraOpen && (
                <div className="flex flex-col items-center space-y-6">
                    <div className="w-full">
                      <div className="relative w-full h-72 bg-black rounded-2xl overflow-hidden shadow-inner flex justify-center mb-6">
                        <video ref={videoRef} autoPlay playsInline muted className="w-full h-full object-cover" />
                      </div>
                      <canvas ref={canvasRef} className="hidden"></canvas>
                      <div className="flex gap-4 w-full">
                        <button onClick={captureFrame} className="flex-1 flex justify-center items-center gap-2 bg-emerald-600 hover:bg-emerald-700 text-white font-semibold py-4 px-6 rounded-xl"><Aperture className="w-5 h-5" /> {t.captureBtn}</button>
                        <button onClick={reset} className="px-6 py-4 rounded-xl font-semibold bg-gray-100">{t.closeCamera}</button>
                      </div>
                    </div>
                </div>
              )}

              {previews.length > 0 && !isCameraOpen && (
                <div className="flex flex-col items-center space-y-6 animate-in fade-in">
                  {!prediction && (
                    <div className="w-full">
                       <div className="flex flex-wrap gap-4 justify-center mb-6">
                         {previews.map((p, i) => (
                            <img key={i} src={p} alt="preview" className="h-32 w-32 object-cover rounded-xl shadow-sm border border-slate-200" />
                         ))}
                       </div>
                    </div>
                  )}

                  {!loading && !prediction && !error && (
                    <div className="flex gap-4 w-full">
                      <button onClick={uploadImage} className="flex-1 bg-emerald-600 text-white py-4 px-6 rounded-xl font-semibold">Analyze {previews.length} Image(s)</button>
                      <button onClick={reset} className="px-6 rounded-xl font-semibold bg-gray-100">{t.cancelBtn}</button>
                    </div>
                  )}

                  {loading && <div className="py-6 text-center"><Loader2 className="w-10 h-10 text-emerald-600 animate-spin mx-auto mb-4" /></div>}

                  {prediction && prediction.status === "conflict" && (
                      <div className="w-full bg-orange-50 border border-orange-200 p-6 rounded-2xl text-center">
                          <AlertCircle className="w-12 h-12 text-orange-400 mx-auto mb-3" />
                          <h3 className="text-orange-900 font-bold text-lg mb-2">Analysis Conflict / Uncertainty</h3>
                          <p className="text-orange-800 mb-6">{prediction.solution}</p>
                          <button onClick={reset} className="bg-orange-600 text-white px-6 py-3 rounded-xl font-semibold">{t.tryAgain}</button>
                      </div>
                  )}

                  {prediction && prediction.status === "stable" && (
                    <div className="w-full">
                      <div className="bg-gradient-to-br from-emerald-50 to-teal-50 p-6 rounded-2xl mb-4 shadow-sm border border-emerald-100">
                         <div className="flex justify-between items-start mb-4">
                            <div className="flex gap-4 items-center">
                              <img src={previews[0]} alt="disease stub" className="w-16 h-16 rounded-lg object-cover border border-emerald-200" />
                              <div>
                                <h3 className="text-xl font-bold text-emerald-900 capitalize">{prediction.final_prediction.replace(/___/g, ' - ').replace(/_/g, ' ')}</h3>
                                <div className="flex items-center gap-2 mt-1 flex-wrap">
                                  <span className="flex items-center gap-1 border border-emerald-200 bg-white px-2 py-0.5 rounded-full text-xs font-bold text-emerald-700">
                                     <CheckCircle className="w-3 h-3 text-emerald-500" /> {prediction.confidence}% {t.validated}
                                  </span>
                                  <span className="flex items-center gap-1 border border-blue-200 bg-white px-2 py-0.5 rounded-full text-xs font-bold text-blue-700">
                                     <AlertCircle className="w-3 h-3 text-blue-500" /> {prediction.cross_image_consistency}% Consensus
                                  </span>
                                </div>
                              </div>
                            </div>
                            <button onClick={handleDownloadReport} disabled={isDownloading} className="flex items-center gap-2 bg-emerald-600 text-white px-4 py-2 rounded-lg text-sm font-semibold hover:bg-emerald-700 shadow-sm">
                              {isDownloading ? <Loader2 className="w-4 h-4 animate-spin" /> : <FileText className="w-4 h-4" />}
                              <span className="hidden sm:inline">{isDownloading ? t.downloadingReport : t.downloadReport}</span>
                            </button>
                         </div>
                        <div className="bg-white p-4 rounded-xl border border-emerald-50 shadow-sm mb-4">
                          <h4 className="font-semibold text-gray-900 mb-1 flex items-center gap-2 text-sm"><Leaf className="w-4 h-4 text-emerald-600" /> {t.remedy}</h4>
                          <p className="text-gray-700 text-sm">{prediction.solution}</p>
                        </div>
                        
                        {prediction.image_results.map((res, idx) => res.cam_base64 && (
                          <div key={idx} className="border border-emerald-100 rounded-xl overflow-hidden bg-white mt-4 flex items-center p-3 gap-4">
                             <div className="w-24 h-24 bg-gray-100 rounded-lg overflow-hidden flex-shrink-0"><img src={res.cam_base64} className="w-full h-full object-cover" /></div>
                             <div className="flex-1">
                                 <h4 className="font-semibold text-gray-900 text-sm"><Zap className="w-4 h-4 inline text-orange-500" /> Frame {idx+1} Heatmap</h4>
                                 <p className="text-xs text-slate-500 mt-1">TTA Consistency: {res.consistency_score}%</p>
                             </div>
                          </div>
                        ))}

                        {/* Phase 13 FEEDBACK UI */}
                        <div className="mt-4 pt-4 border-t border-emerald-200/50">
                            {feedbackState === 'none' && (
                                <div className="flex items-center gap-3">
                                    <span className="text-sm font-semibold text-emerald-800 flex-1">Is this prediction accurate?</span>
                                    <button onClick={() => submitFeedback(true)} className="flex items-center gap-1 bg-white border border-emerald-200 px-3 py-1.5 rounded-lg text-sm text-emerald-700 hover:bg-emerald-50"><ThumbsUp className="w-4 h-4"/> {t.feedbackCorrect}</button>
                                    <button onClick={() => setFeedbackState('wrong')} className="flex items-center gap-1 bg-white border border-red-200 px-3 py-1.5 rounded-lg text-sm text-red-700 hover:bg-red-50"><ThumbsDown className="w-4 h-4"/> {t.feedbackWrong}</button>
                                </div>
                            )}
                            {feedbackState === 'wrong' && (
                                <div className="flex items-center gap-3">
                                    <select className="flex-1 text-sm border-gray-300 rounded-lg p-2" value={selectedCorrectLabel} onChange={(e) => setSelectedCorrectLabel(e.target.value)}>{AVAILABLE_LABELS.map(lbl => <option key={lbl} value={lbl}>{lbl}</option>)}</select>
                                    <button onClick={() => submitFeedback(false)} className="bg-emerald-600 text-white px-4 py-2 rounded-lg text-sm font-semibold hover:bg-emerald-700">{t.feedbackSubmit}</button>
                                    <button onClick={() => setFeedbackState('none')} className="text-gray-500 hover:text-gray-700"><X className="w-5 h-5"/></button>
                                </div>
                            )}
                            {feedbackState === 'submitted' && <p className="text-sm font-semibold text-emerald-700 flex items-center gap-2"><CheckCircle className="w-4 h-4"/> {t.feedbackThanks}</p>}
                        </div>
                      </div>

                      <button onClick={reset} className="w-full bg-white border-2 border-slate-200 text-slate-700 font-semibold py-4 px-6 rounded-xl flex justify-center items-center gap-2 mb-4">
                        <UploadCloud className="w-5 h-5" /> {t.scanAnother}
                      </button>
                    </div>
                  )}
                </div>
              )}
            </div>
          </main>
      )}
      {/* AI AGRONOMIST CHAT OVERLAY */}
      {prediction && prediction.status === "stable" && !isDashboard && (
        <div className="fixed bottom-6 right-6 w-96 max-h-[500px] bg-white shadow-2xl rounded-2xl border border-emerald-100 flex flex-col z-50 animate-in slide-in-from-bottom-6 group">
            <div className="bg-emerald-600 p-4 rounded-t-2xl flex items-center justify-between text-white shadow-lg">
                <div className="flex items-center gap-2">
                    <Bot className="w-5 h-5" />
                    <h3 className="font-bold text-sm tracking-tight">{t.chatTitle}</h3>
                </div>
                <div className="flex items-center gap-1">
                    <span className="w-2 h-2 bg-emerald-300 rounded-full animate-pulse"></span>
                    <span className="text-[10px] font-bold uppercase opacity-80">Live</span>
                </div>
            </div>
            
            <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-slate-50/50 min-h-[300px]">
                {chatMessages.length === 0 && (
                    <div className="h-full flex flex-col items-center justify-center text-center p-6 space-y-2 opacity-60">
                        <MessageSquare className="w-8 h-8 text-emerald-400" />
                        <p className="text-xs font-medium text-slate-500">{t.chatHelpText}</p>
                    </div>
                )}
                {chatMessages.map(msg => (
                    <div key={msg.id} className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'} animate-in fade-in`}>
                        <div className={`max-w-[85%] p-3 rounded-2xl text-sm shadow-sm ${msg.sender === 'user' ? 'bg-emerald-600 text-white rounded-br-none' : 'bg-white border border-slate-200 text-slate-800 rounded-bl-none'}`}>
                            {msg.text}
                        </div>
                    </div>
                ))}
                {isChatLoading && (
                    <div className="flex justify-start">
                        <div className="bg-white border border-slate-200 p-3 rounded-2xl rounded-bl-none shadow-sm">
                            <Loader2 className="w-4 h-4 text-emerald-500 animate-spin" />
                        </div>
                    </div>
                )}
                <div ref={chatBottomRef} />
            </div>

            <form onSubmit={sendChatMessage} className="p-3 border-t border-slate-100 bg-white rounded-b-2xl">
                <div className="relative flex items-center">
                    <input 
                        type="text" 
                        value={chatInput} 
                        onChange={(e) => setChatInput(e.target.value)} 
                        placeholder={t.askBotPlaceholder}
                        className="w-full bg-slate-100 border-none rounded-xl py-3 pl-4 pr-12 text-sm focus:ring-2 focus:ring-emerald-500 transition-all"
                    />
                    <button type="submit" disabled={!chatInput.trim() || isChatLoading} className="absolute right-2 p-2 bg-emerald-600 text-white rounded-lg hover:bg-emerald-700 disabled:opacity-50 transition-colors">
                        <Send className="w-4 h-4" />
                    </button>
                </div>
            </form>
        </div>
      )}
    </div>
  );
}

export default App;
