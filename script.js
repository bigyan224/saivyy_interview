import { TalkingHead } from "talkinghead";

// Element References
const videoElement = document.getElementById('videoElement');
const startButton = document.getElementById('startButton');
const topicInput = document.getElementById('topicInput');
const loadingElement = document.getElementById('loading');
const interviewContainer = document.getElementById('interviewContainer');
const setupContainer = document.getElementById('setupContainer');
const chatContainer = document.getElementById('chatContainer');
const userInput = document.getElementById('userInput');
const sendButton = document.getElementById('sendButton');
const micButton = document.getElementById('micButton');
const statusIndicator = document.getElementById('statusIndicator');
const outfitAnalysisContainer = document.getElementById('outfitAnalysisContainer');
const closeOutfitAnalysis = document.getElementById('closeOutfitAnalysis');
const outfitAnalysisToggle = document.getElementById('outfitAnalysisToggle');
const expressionAnalysisContainer = document.getElementById('expressionAnalysisContainer');
const closeExpressionAnalysis = document.getElementById('closeExpressionAnalysis');
const expressionAnalysisToggle = document.getElementById('expressionAnalysisToggle');
const expressionAnalysisContent = document.getElementById('expressionAnalysisContent');
const expressionSummary = document.getElementById('expressionSummary');
const canvasOutput = document.getElementById('canvasOutput');
const endInterviewButton = document.getElementById('endInterviewButton');
// --- Live Chat Elements ---
const liveChatToggle = document.getElementById('liveChatToggle');
const liveChatStatus = document.getElementById('liveChatStatus');
// --- End Live Chat Elements ---

// State variables
let ws = null;
let captureInterval = null;
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;
let speechSynthesis = window.speechSynthesis;
let recognition = null; // For regular mic button
let head;
let outfitAnalysisData = null;

// --- Live Chat State ---
let isLiveChatActive = false;
let liveChatRecognition = null;
let silenceTimer = null;
const SILENCE_DELAY = 2000; // 2 seconds
let liveFinalTranscript = '';
// --- End Live Chat State ---

// Facial expression variables
let cv = null;
let faceClassifier = null;
let eyeClassifier = null;
let faceMatcher = null;
let expressionInterval = null;
let expressionCount = 0;
let expressionData = {
    smile: 0,
    neutral: 0,
    frown: 0,
    lookAway: 0,
    blink: 0,
    samples: 0
};
let expressionHistory = [];
let isExpressionAnalysisActive = false;

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    setupCamera();
    setupSpeechRecognition();
    initializeAvatar();
    setupEventListeners();
    
    // No need to initialize OpenCV anymore, server will handle this
    statusIndicator.textContent = 'Ready to start your interview - using server-side expression analysis';
});

// Initialize OpenCV.js
function initOpenCV() {
    console.log("Initializing OpenCV...");
    
    // Check if OpenCV is already initialized
    if (typeof cv !== 'undefined' && cv) {
        console.log("OpenCV.js already available");
        cv = window.cv;
        loadFaceDetectionModels();
        return;
    }
    
    // Listen for the OpenCV ready event
    window.addEventListener('opencv-ready', function() {
        console.log("OpenCV ready event received");
        if (typeof cv !== 'undefined' && cv) {
            cv = window.cv;
            loadFaceDetectionModels();
        }
    });
    
    // Listen for OpenCV load failure
    window.addEventListener('opencv-load-failed', function() {
        console.log("OpenCV load failed event received, switching to fallback mode");
        expressionAnalysisToggle.style.display = 'flex';
        statusIndicator.textContent = 'Using simplified expression analysis (OpenCV unavailable)';
    });
    
    // Poll for OpenCV availability
    let attempts = 0;
    const maxAttempts = 10;
    const checkInterval = setInterval(() => {
        attempts++;
        console.log(`Checking for OpenCV (attempt ${attempts}/${maxAttempts})...`);
        
        if (typeof cv !== 'undefined' && cv) {
            clearInterval(checkInterval);
            console.log("OpenCV.js detected through polling");
            cv = window.cv;
            loadFaceDetectionModels();
        } else if (attempts >= maxAttempts) {
            clearInterval(checkInterval);
            console.error("OpenCV initialization failed after maximum attempts");
            expressionAnalysisToggle.style.display = 'none';
            statusIndicator.textContent = 'Using simplified expression analysis (OpenCV unavailable)';
        }
    }, 1000);
    
    // Ensure the script is in the document
    if (!document.querySelector('script[src*="opencv.js"]')) {
        console.log("OpenCV script not found, adding it dynamically");
        const script = document.createElement('script');
        script.async = true;
        script.src = 'https://docs.opencv.org/master/opencv.js';
        script.onload = function() {
            console.log("OpenCV script loaded dynamically");
        };
        script.onerror = function() {
            console.error("Failed to load OpenCV.js script");
            clearInterval(checkInterval);
            expressionAnalysisToggle.style.display = 'none';
            statusIndicator.textContent = 'Expression analysis unavailable - OpenCV failed to load';
        };
        document.head.appendChild(script);
    }
}

// Load face detection models
async function loadFaceDetectionModels() {
    if (!cv) {
        console.error("OpenCV not initialized yet");
        return;
    }
    
    try {
        // Load face classifier
        faceClassifier = new cv.CascadeClassifier();
        
        // Use a more reliable method to load the XML cascade files
        const faceCascadeUrl = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml';
        const eyeCascadeUrl = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml';
        
        // Utility function to fetch XML and convert to Uint8Array
        const fetchCascade = async (url) => {
            const response = await fetch(url);
            if (!response.ok) {
                throw new Error(`Failed to fetch cascade from ${url}`);
            }
            const text = await response.text();
            const data = new Uint8Array(text.length);
            for (let i = 0; i < text.length; ++i) {
                data[i] = text.charCodeAt(i);
            }
            return data;
        };
        
        // Load face cascade
        const faceCascadeData = await fetchCascade(faceCascadeUrl);
        
        // Create file in memory
        const faceMat = cv.matFromArray(1, faceCascadeData.length, cv.CV_8UC1, faceCascadeData);
        const faceMatVec = new cv.MatVector();
        faceMatVec.push_back(faceMat);
        
        // Load classifier from memory
        faceClassifier.load(faceMatVec);
        
        // Load eye classifier
        eyeClassifier = new cv.CascadeClassifier();
        const eyeCascadeData = await fetchCascade(eyeCascadeUrl);
        
        const eyeMat = cv.matFromArray(1, eyeCascadeData.length, cv.CV_8UC1, eyeCascadeData);
        const eyeMatVec = new cv.MatVector();
        eyeMatVec.push_back(eyeMat);
        
        eyeClassifier.load(eyeMatVec);
        
        // Clean up
        faceMat.delete();
        faceMatVec.delete();
        eyeMat.delete();
        eyeMatVec.delete();
        
        console.log("Face detection models loaded successfully");
        expressionAnalysisToggle.style.display = 'flex';
        
        // Start a test analysis to verify everything works
        setTimeout(analyzeExpressions, 1000);
        
    } catch (error) {
        console.error("Error loading face detection models:", error);
        expressionAnalysisToggle.style.display = 'none';
    }
}

// Analyze facial expressions
function analyzeExpressions() {
    if (!cv || !faceClassifier || !eyeClassifier) {
        console.log("OpenCV or classifiers not initialized yet");
        return;
    }
    
    const cameraElement = document.getElementById('cameraElement');
    if (!cameraElement || !cameraElement.videoWidth || !cameraElement.videoHeight) {
        console.log("Camera not ready yet");
        return;
    }
    
    try {
        // Set up canvas for processing
        const canvasContext = canvasOutput.getContext('2d');
        canvasOutput.width = cameraElement.videoWidth;
        canvasOutput.height = cameraElement.videoHeight;
        
        // Draw current frame to canvas
        canvasContext.drawImage(cameraElement, 0, 0, canvasOutput.width, canvasOutput.height);
        
        // Create OpenCV matrices
        const src = new cv.Mat(canvasOutput.height, canvasOutput.width, cv.CV_8UC4);
        const gray = new cv.Mat();
        const faces = new cv.RectVector();
        const eyes = new cv.RectVector();
        
        // Get image data from canvas and copy to OpenCV matrix
        const imgData = canvasContext.getImageData(0, 0, canvasOutput.width, canvasOutput.height);
        src.data.set(imgData.data);
        
        // Convert to grayscale for better detection
        cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
        
        // Detect faces
        faceClassifier.detectMultiScale(gray, faces, 1.1, 3, 0, new cv.Size(50, 50));
        
        // Initialize scores
        let smileScore = 0;
        let eyeOpenScore = 0;
        let lookAwayScore = 0;
        
        // Process faces if any found
        if (faces.size() > 0) {
            // Use first face detected (assuming single user)
            const face = faces.get(0);
            const faceRegion = new cv.Rect(face.x, face.y, face.width, face.height);
            
            // Calculate how centered the face is
            const centerX = canvasOutput.width / 2;
            const faceCenter = face.x + (face.width / 2);
            const distanceFromCenter = Math.abs(faceCenter - centerX) / (canvasOutput.width / 2);
            
            // If face is not centered, count as looking away
            lookAwayScore = distanceFromCenter > 0.2 ? 1 : 0;
            
            // Get region of interest (just the face)
            const faceROI = gray.roi(faceRegion);
            
            // Detect eyes within the face region
            eyeClassifier.detectMultiScale(faceROI, eyes);
            eyeOpenScore = eyes.size() >= 2 ? 1 : 0;
            
            // Improved smile detection - using aspect ratio and position
            // A smiling face tends to be wider relative to height
            // This is a simplified approach - real smile detection would use more advanced features
            const faceRatio = face.width / face.height;
            
            // Calculate position in lower half of face (mouth region)
            let lowerFaceROI = new cv.Mat();
            const lowerFaceRect = new cv.Rect(
                0, 
                Math.floor(faceROI.rows * 0.6), 
                faceROI.cols,
                Math.floor(faceROI.rows * 0.4)
            );
            
            try {
                // Extract lower face region for smile detection
                lowerFaceROI = faceROI.roi(lowerFaceRect);
                
                // Simple smile detection based on face ratio and aspect
                // More accurate smile detection would use machine learning or more complex features
                smileScore = faceRatio > 1.35 ? 0.7 : 0;
                
                // Clean up
                lowerFaceROI.delete();
            } catch (e) {
                console.error("Error processing lower face:", e);
            }
            
            // Clean up face ROI
            faceROI.delete();
        } else {
            // No face detected, count as looking away
            lookAwayScore = 1;
        }
        
        // Update expression counts for this frame
        expressionData.samples++;
        expressionData.smile += smileScore;
        expressionData.neutral += (1 - smileScore) * (1 - lookAwayScore);
        expressionData.frown += (1 - smileScore) * (1 - lookAwayScore) * 0.2; // Approximate frown
        expressionData.lookAway += lookAwayScore;
        expressionData.blink += (1 - eyeOpenScore);
        
        // Calculate and record statistics every 5 seconds
        expressionCount++;
        if (expressionCount >= 50) { // 10fps * 5 seconds
            expressionCount = 0;
            
            // Calculate percentages for this interval
            const intervalData = {
                timestamp: new Date(),
                smile: (expressionData.smile / expressionData.samples) * 100,
                neutral: (expressionData.neutral / expressionData.samples) * 100,
                frown: (expressionData.frown / expressionData.samples) * 100,
                lookAway: (expressionData.lookAway / expressionData.samples) * 100,
                blink: (expressionData.blink / expressionData.samples) * 100
            };
            
            // Add to history and update display
            expressionHistory.push(intervalData);
            updateExpressionAnalysis(intervalData);
            
            // Reset counters for next interval
            expressionData.smile = 0;
            expressionData.neutral = 0;
            expressionData.frown = 0;
            expressionData.lookAway = 0;
            expressionData.blink = 0;
            expressionData.samples = 0;
        }
        
        // Clean up OpenCV resources
        src.delete();
        gray.delete();
        faces.delete();
        eyes.delete();
        
    } catch (err) {
        console.error('Error in expression analysis:', err);
    }
}

// Update expression analysis display
function updateExpressionAnalysis(data) {
    const entry = document.createElement('div');
    entry.className = 'expression-entry';
    
    const time = document.createElement('div');
    time.className = 'expression-time';
    time.textContent = `${data.timestamp.toLocaleTimeString()}`;
    
    const details = document.createElement('div');
    details.className = 'expression-details';
    
    // Add expression metrics
    const expressions = [
        { name: 'Smile', value: Math.round(data.smile), icon: 'fa-smile' },
        { name: 'Neutral', value: Math.round(data.neutral), icon: 'fa-meh' },
        { name: 'Frown', value: Math.round(data.frown), icon: 'fa-frown' },
        { name: 'Look Away', value: Math.round(data.lookAway), icon: 'fa-eye-slash' },
        { name: 'Blink', value: Math.round(data.blink), icon: 'fa-low-vision' }
    ];
    
    expressions.forEach(exp => {
        const item = document.createElement('div');
        item.className = 'expression-item';
        
        const label = document.createElement('div');
        label.className = 'expression-item-label';
        label.innerHTML = `<i class="fas ${exp.icon}"></i> ${exp.name}`;
        
        const value = document.createElement('div');
        value.className = 'expression-item-value';
        value.textContent = `${exp.value}%`;
        
        item.appendChild(label);
        item.appendChild(value);
        details.appendChild(item);
    });
    
    // Add suggestion based on expression data
    const suggestion = document.createElement('div');
    suggestion.className = 'expression-suggestion';
    
    if (data.lookAway > 30) {
        suggestion.textContent = 'Try to maintain more eye contact with the interviewer.';
    } else if (data.neutral > 70) {
        suggestion.textContent = 'Consider showing more expressiveness. A slight smile can help establish rapport.';
    } else if (data.smile > 70) {
        suggestion.textContent = 'Your smile is great, but mix in some neutral expressions for a balanced presentation.';
    } else if (data.frown > 20) {
        suggestion.textContent = 'Be mindful of frowning, as it may convey negative emotions or confusion.';
    } else if (data.blink > 30) {
        suggestion.textContent = 'You may be blinking more than usual, which can indicate nervousness. Try to relax.';
    } else {
        suggestion.textContent = 'Your expressions look balanced and appropriate for an interview setting.';
    }
    
    entry.appendChild(time);
    entry.appendChild(details);
    entry.appendChild(suggestion);
    
    expressionAnalysisContent.prepend(entry);
    
    // Keep only the last 10 entries to avoid excessive memory usage
    const entries = expressionAnalysisContent.querySelectorAll('.expression-entry');
    if (entries.length > 10) {
        for (let i = 10; i < entries.length; i++) {
            entries[i].remove();
        }
    }
}

// Generate expression summary
function generateExpressionSummary() {
    console.log("Generating expression summary...");
    
    // Check if we have any data to analyze
    if (expressionHistory.length === 0) {
        expressionSummary.innerHTML = `
            <h3>Interview Expression Analysis</h3>
            <p>No expression data was collected during this interview. Please ensure your camera is enabled and your face is visible during future interviews.</p>
        `;
        return;
    }
    
    // Calculate averages for each expression
    let totalSmile = 0, totalNeutral = 0, totalFrown = 0, totalLookAway = 0, totalBlink = 0;
    
    expressionHistory.forEach(data => {
        totalSmile += data.smile;
        totalNeutral += data.neutral;
        totalFrown += data.frown;
        totalLookAway += data.lookAway;
        totalBlink += data.blink;
    });
    
    const count = expressionHistory.length;
    const avgSmile = (totalSmile / count).toFixed(1);
    const avgNeutral = (totalNeutral / count).toFixed(1);
    const avgFrown = (totalFrown / count).toFixed(1);
    const avgLookAway = (totalLookAway / count).toFixed(1);
    const avgBlink = (totalBlink / count).toFixed(1);
    
    // Calculate trend over time (comparing first 1/3 vs last 1/3)
    const firstThird = Math.floor(count / 3);
    const lastThird = Math.max(firstThird, count - firstThird);
    
    let earlySmile = 0, earlyNeutral = 0, earlyFrown = 0, earlyLookAway = 0, earlyBlink = 0;
    let lateSmile = 0, lateNeutral = 0, lateFrown = 0, lateLookAway = 0, lateBlink = 0;
    
    // Calculate early averages
    for (let i = 0; i < firstThird; i++) {
        earlySmile += expressionHistory[i].smile;
        earlyNeutral += expressionHistory[i].neutral;
        earlyFrown += expressionHistory[i].frown;
        earlyLookAway += expressionHistory[i].lookAway;
        earlyBlink += expressionHistory[i].blink;
    }
    
    // Calculate late averages
    for (let i = lastThird; i < count; i++) {
        lateSmile += expressionHistory[i].smile;
        lateNeutral += expressionHistory[i].neutral;
        lateFrown += expressionHistory[i].frown;
        lateLookAway += expressionHistory[i].lookAway;
        lateBlink += expressionHistory[i].blink;
    }
    
    earlySmile /= firstThird;
    earlyNeutral /= firstThird;
    earlyFrown /= firstThird;
    earlyLookAway /= firstThird;
    earlyBlink /= firstThird;
    
    lateSmile /= (count - lastThird);
    lateNeutral /= (count - lastThird);
    lateFrown /= (count - lastThird);
    lateLookAway /= (count - lastThird);
    lateBlink /= (count - lastThird);
    
    // Calculate percentage changes
    const smileChange = ((lateSmile - earlySmile) / earlySmile * 100).toFixed(1);
    const neutralChange = ((lateNeutral - earlyNeutral) / earlyNeutral * 100).toFixed(1);
    const frownChange = ((lateFrown - earlyFrown) / earlyFrown * 100).toFixed(1);
    const lookAwayChange = ((lateLookAway - earlyLookAway) / earlyLookAway * 100).toFixed(1);
    const blinkChange = ((lateBlink - earlyBlink) / earlyBlink * 100).toFixed(1);
    
    // Determine engagement level
    let engagementScore = 0;
    engagementScore += parseFloat(avgSmile) * 0.3; // Smiling is good for engagement
    engagementScore -= parseFloat(avgFrown) * 0.4; // Frowning may indicate confusion
    engagementScore -= parseFloat(avgLookAway) * 0.5; // Looking away indicates disengagement
    engagementScore -= parseFloat(avgBlink) * 0.1; // Excessive blinking may indicate nervousness
    engagementScore += (parseFloat(smileChange) > 0 ? 5 : -5); // Increasing smile is good
    engagementScore -= (parseFloat(lookAwayChange) > 0 ? 5 : 0); // Increasing looking away is bad
    
    let engagementLevel = 'Average';
    let engagementColor = '#FFA500'; // Orange
    
    if (engagementScore > 20) {
        engagementLevel = 'Excellent';
        engagementColor = '#008000'; // Green
    } else if (engagementScore > 10) {
        engagementLevel = 'Good';
        engagementColor = '#90EE90'; // Light green
    } else if (engagementScore < -15) {
        engagementLevel = 'Poor';
        engagementColor = '#FF0000'; // Red
    } else if (engagementScore < -5) {
        engagementLevel = 'Needs improvement';
        engagementColor = '#FFA07A'; // Light salmon
    }
    
    // Generate personalized recommendations
    let recommendations = [];
    
    if (parseFloat(avgSmile) < 20) {
        recommendations.push("Try to smile more during interviews to create a positive impression. Even a slight smile can make you appear more engaging and confident.");
    }
    
    if (parseFloat(avgLookAway) > 25) {
        recommendations.push("Work on maintaining better eye contact. Looking away frequently can give an impression of disinterest or lack of confidence.");
    }
    
    if (parseFloat(avgFrown) > 10) {
        recommendations.push("Be mindful of your facial expressions. You appear to frown or look confused at times, which might give interviewers the wrong impression about your understanding or confidence.");
    }
    
    if (parseFloat(avgBlink) > 20) {
        recommendations.push("Your blink rate suggests you might have been nervous. Practice relaxation techniques before interviews to appear more calm and composed.");
    }
    
    if (parseFloat(smileChange) < -15) {
        recommendations.push("You smiled less as the interview progressed. Try to maintain your energy and positive expressions throughout the entire interview.");
    }
    
    if (parseFloat(lookAwayChange) > 15) {
        recommendations.push("Your eye contact decreased during the interview. Make a conscious effort to maintain engagement even when the interview gets challenging.");
    }
    
    if (recommendations.length === 0) {
        recommendations.push("Your expression patterns appear good. Continue to practice maintaining appropriate expressions during your interviews.");
    }
    
    // Create the HTML content for the summary
    let htmlContent = `
        <h3>Interview Expression Analysis</h3>
        <p>This analysis is based on ${count} data points collected during your interview.</p>
        
        <div class="summary-section">
            <h4>Expression Averages</h4>
            <div class="expression-bars">
                <div class="expression-bar">
                    <div class="bar-label">Smile</div>
                    <div class="bar-container">
                        <div class="bar-fill" style="width: ${Math.min(100, avgSmile)}%; background-color: #4CAF50;"></div>
                    </div>
                    <div class="bar-value">${avgSmile}%</div>
                </div>
                <div class="expression-bar">
                    <div class="bar-label">Neutral</div>
                    <div class="bar-container">
                        <div class="bar-fill" style="width: ${Math.min(100, avgNeutral)}%; background-color: #2196F3;"></div>
                    </div>
                    <div class="bar-value">${avgNeutral}%</div>
                </div>
                <div class="expression-bar">
                    <div class="bar-label">Frown</div>
                    <div class="bar-container">
                        <div class="bar-fill" style="width: ${Math.min(100, avgFrown)}%; background-color: #FF9800;"></div>
                    </div>
                    <div class="bar-value">${avgFrown}%</div>
                </div>
                <div class="expression-bar">
                    <div class="bar-label">Looking Away</div>
                    <div class="bar-container">
                        <div class="bar-fill" style="width: ${Math.min(100, avgLookAway)}%; background-color: #F44336;"></div>
                    </div>
                    <div class="bar-value">${avgLookAway}%</div>
                </div>
                <div class="expression-bar">
                    <div class="bar-label">Blinking</div>
                    <div class="bar-container">
                        <div class="bar-fill" style="width: ${Math.min(100, avgBlink)}%; background-color: #9C27B0;"></div>
                    </div>
                    <div class="bar-value">${avgBlink}%</div>
                </div>
            </div>
        </div>
        
        <div class="summary-section">
            <h4>Expression Trends</h4>
            <p>Changes in your expressions from the beginning to the end of the interview:</p>
            <ul>
                <li>Smile: <span class="${smileChange > 0 ? 'positive' : 'negative'}">${smileChange > 0 ? '+' : ''}${smileChange}%</span></li>
                <li>Neutral: <span class="${neutralChange > 0 ? 'positive' : 'negative'}">${neutralChange > 0 ? '+' : ''}${neutralChange}%</span></li>
                <li>Frown: <span class="${frownChange < 0 ? 'positive' : 'negative'}">${frownChange > 0 ? '+' : ''}${frownChange}%</span></li>
                <li>Looking Away: <span class="${lookAwayChange < 0 ? 'positive' : 'negative'}">${lookAwayChange > 0 ? '+' : ''}${lookAwayChange}%</span></li>
                <li>Blinking: <span class="${blinkChange < 0 ? 'positive' : 'negative'}">${blinkChange > 0 ? '+' : ''}${blinkChange}%</span></li>
            </ul>
        </div>
        
        <div class="summary-section">
            <h4>Overall Engagement Level</h4>
            <p>Your overall engagement score: <span style="color: ${engagementColor}; font-weight: bold;">${engagementLevel}</span></p>
        </div>
        
        <div class="summary-section">
            <h4>Personalized Recommendations</h4>
            <ul>
                ${recommendations.map(rec => `<li>${rec}</li>`).join('')}
            </ul>
        </div>
    `;
    
    // Update the summary element
    expressionSummary.innerHTML = htmlContent;
    
    // Add styles for the new elements
    const style = document.createElement('style');
    style.textContent = `
        .summary-section {
            margin-bottom: 20px;
            background-color: #f5f5f5;
            padding: 15px;
            border-radius: 8px;
        }
        
        .expression-bars {
            margin-top: 10px;
        }
        
        .expression-bar {
            display: flex;
            align-items: center;
            margin-bottom: 8px;
        }
        
        .bar-label {
            width: 100px;
            font-weight: 500;
        }
        
        .bar-container {
            flex-grow: 1;
            height: 20px;
            background-color: #e0e0e0;
            border-radius: 4px;
            overflow: hidden;
        }
        
        .bar-fill {
            height: 100%;
            transition: width 1s ease-in-out;
        }
        
        .bar-value {
            width: 50px;
            text-align: right;
            margin-left: 10px;
            font-weight: 500;
        }
        
        .positive {
            color: green;
            font-weight: 500;
        }
        
        .negative {
            color: red;
            font-weight: 500;
        }
    `;
    
    document.head.appendChild(style);
}

// Toggle expression analysis
function toggleExpressionAnalysis(start = true) {
    if (start && !isExpressionAnalysisActive) {
        // Start analysis
        isExpressionAnalysisActive = true;
        
        // Use server-side analysis instead of local OpenCV
        if (ws && ws.readyState === WebSocket.OPEN) {
            // Notify server to start expression analysis
            ws.send(JSON.stringify({
                type: 'start_expression_analysis'
            }));
            
            // Start sending camera frames to the server
            expressionInterval = setInterval(captureAndSendExpressionFrame, 100); // 10fps analysis
            statusIndicator.textContent = 'Expression analysis active (Server-side processing)';
        } else {
            statusIndicator.textContent = 'Expression analysis unavailable: No server connection';
            isExpressionAnalysisActive = false;
            return;
        }
        
        expressionAnalysisToggle.style.display = 'flex';
    } else if (!start && isExpressionAnalysisActive) {
        // Stop analysis
        isExpressionAnalysisActive = false;
        clearInterval(expressionInterval);
        expressionInterval = null;
        
        // Notify server to stop expression analysis
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({
                type: 'stop_expression_analysis'
            }));
        }
        
        statusIndicator.textContent = 'Expression analysis stopped';
    }
}

// Capture and send frame for expression analysis
function captureAndSendExpressionFrame() {
    if (!isExpressionAnalysisActive || !ws || ws.readyState !== WebSocket.OPEN) {
        return;
    }
    
    const canvas = document.createElement('canvas');
    const cameraElement = document.getElementById('cameraElement');
    canvas.width = cameraElement.videoWidth;
    canvas.height = cameraElement.videoHeight;
    
    if (canvas.width === 0 || canvas.height === 0) {
        console.error('Invalid canvas dimensions for expression analysis:', canvas.width, canvas.height);
        return;
    }
    
    const ctx = canvas.getContext('2d');
    ctx.drawImage(cameraElement, 0, 0, canvas.width, canvas.height);
    
    try {
        const imageData = canvas.toDataURL('image/jpeg', 0.7); // Lower quality for faster transmission
        if (!imageData || imageData.length < 22) { // Minimum length for a valid data URL
            console.error('Invalid image data generated for expression analysis');
            return;
        }
        
        const base64Data = imageData.split(',')[1];
        if (!base64Data) {
            console.error('Failed to extract base64 data for expression analysis');
            return;
        }
        
        // Send to server for analysis
        ws.send(JSON.stringify({
            type: 'expression_frame',
            data: base64Data
        }));
    } catch (error) {
        console.error('Error capturing frame for expression analysis:', error);
    }
}

// Set up camera access
async function setupCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        const cameraElement = document.getElementById('cameraElement');
        cameraElement.srcObject = stream;
        startButton.disabled = false;
    } catch (err) {
        console.error('Error accessing camera:', err);
        statusIndicator.textContent = 'Error: Camera access denied';
        alert('Error accessing camera. Please make sure you have granted camera permissions.');
    }
}

// Setup speech recognition for both regular mic and live chat
function setupSpeechRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
        console.error('Speech Recognition API not supported in this browser.');
        micButton.disabled = true;
        liveChatToggle.disabled = true; // Disable live chat too
        micButton.title = 'Speech recognition not supported';
        liveChatToggle.title = 'Speech recognition not supported';
        statusIndicator.textContent = 'Speech recognition not supported.';
        return;
    }

    // --- Setup for regular mic button ---
    recognition = new SpeechRecognition();
    recognition.continuous = false; // Stop after first utterance
    recognition.interimResults = false;

    recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        userInput.value = transcript;
        stopRecording(); // Visual cue
        sendAnswer(); // Send immediately after recognized
    };

    recognition.onerror = (event) => {
        console.error('Speech recognition error:', event.error);
        statusIndicator.textContent = `Speech error: ${event.error}`;
        stopRecording(); // Visual cue
    };

    recognition.onend = () => {
        if (isRecording) { // Check if it stopped unexpectedly
            stopRecording();
        }
    };
    // --- End setup for regular mic ---


    // --- Setup for Live Chat ---
    liveChatRecognition = new SpeechRecognition();
    liveChatRecognition.continuous = true; // Keep listening
    liveChatRecognition.interimResults = true; // Show results as they come

    liveChatRecognition.onresult = (event) => {
        clearTimeout(silenceTimer); // Clear any existing silence timer

        let interimTranscript = '';
        for (let i = event.resultIndex; i < event.results.length; ++i) {
            if (event.results[i].isFinal) {
                liveFinalTranscript += event.results[i][0].transcript + ' '; // Append final results
            } else {
                interimTranscript += event.results[i][0].transcript;
            }
        }

        // Display interim results or the start of the final transcript
        userInput.value = liveFinalTranscript + interimTranscript;
        userInput.scrollTop = userInput.scrollHeight; // Keep current text visible

        // Start timer to detect silence after the last result
        silenceTimer = setTimeout(() => {
            checkForSilenceAndSend();
        }, SILENCE_DELAY);
    };

    liveChatRecognition.onerror = (event) => {
        console.error('Live Chat recognition error:', event.error);
        // Don't stop live chat on minor errors like 'no-speech', just log it
        if (event.error !== 'no-speech' && event.error !== 'aborted') {
             statusIndicator.textContent = `Live Chat error: ${event.error}`;
             // Consider stopping live chat on critical errors?
             // toggleLiveChat(false); // Optionally turn off live chat on critical errors
        }
        // The 'onend' handler will attempt to restart if isLiveChatActive is true
    };

    liveChatRecognition.onend = () => {
        console.log('Live Chat Recognition ended.');
        // If live chat is supposed to be active, restart it
        // This handles cases where it stops automatically after long silence or network issues
        if (isLiveChatActive) {
            console.log('Restarting Live Chat Recognition...');
            // Small delay before restarting to avoid potential rapid restart issues
            setTimeout(() => {
                if (isLiveChatActive) { // Double check state
                   try {
                       liveChatRecognition.start();
                   } catch (e) {
                       console.error("Error restarting live chat recognition:", e);
                       toggleLiveChat(false); // Turn off if restart fails
                       statusIndicator.textContent = "Live Chat failed to restart.";
                   }
                }
            }, 250);
        }
    };
    // --- End setup for Live Chat ---

    console.log('Speech recognition initialized.');
}

// --- Live Chat Functions ---
function toggleLiveChat(forceState = null) {
    const newState = forceState !== null ? forceState : !isLiveChatActive;

    if (newState) {
        // Activate Live Chat
        isLiveChatActive = true;
        liveChatToggle.classList.add('active');
        liveChatStatus.style.display = 'inline';
        micButton.disabled = true; // Disable regular mic
        micButton.style.opacity = '0.5';
        liveFinalTranscript = ''; // Clear transcript
        userInput.placeholder = "Live Chat active: Start speaking...";
        try {
            console.log("Starting Live Chat Recognition...");
            liveChatRecognition.start();
        } catch(e) {
            console.error("Error starting live chat:", e);
            statusIndicator.textContent = "Could not start live chat.";
            isLiveChatActive = false; // Revert state
            liveChatToggle.classList.remove('active');
            liveChatStatus.style.display = 'none';
            micButton.disabled = false;
            micButton.style.opacity = '1';
            userInput.placeholder = "Type your answer here...";
        }

    } else {
        // Deactivate Live Chat
        isLiveChatActive = false;
        liveChatToggle.classList.remove('active');
        liveChatStatus.style.display = 'none';
        micButton.disabled = false; // Re-enable regular mic
        micButton.style.opacity = '1';
        userInput.placeholder = "Type your answer here...";
        clearTimeout(silenceTimer);
        try {
            liveChatRecognition.stop();
            console.log("Stopped Live Chat Recognition.");
        } catch(e) {
            console.error("Error stopping live chat:", e);
        }
        // Clear any lingering transcript from live chat
        if (userInput.value === liveFinalTranscript || userInput.placeholder === "Live Chat active: Start speaking...") {
            // Avoid clearing user's manually typed text
           // userInput.value = ''; // Optional: Clear input on disable? Debatable UX.
        }
        liveFinalTranscript = '';
    }
}

function checkForSilenceAndSend() {
    if (!isLiveChatActive) return; // Don't send if mode was turned off

    const transcriptToSend = liveFinalTranscript.trim();
    if (transcriptToSend) {
        console.log('Silence detected, sending:', transcriptToSend);
        userInput.value = transcriptToSend; // Ensure final transcript is in the box
        sendAnswer();
        liveFinalTranscript = ''; // Clear transcript after sending
        // Restart listening happens via the 'onend' handler
    } else {
        console.log('Silence detected, but no final transcript captured yet.');
        // Keep listening - onend handler will restart if needed
    }
}
// --- End Live Chat Functions ---

// Initialize Talking Head avatar
async function initializeAvatar() {
    const nodeAvatar = document.getElementById('avatar');
    const nodeLoading = document.getElementById('avatarLoading');
    
    try {
        head = new TalkingHead(nodeAvatar, {
            ttsEndpoint: "https://eu-texttospeech.googleapis.com/v1beta1/text:synthesize",
            ttsApikey: "AIzaSyB8EfzcDlOA-Owp_UI9IY5U6RKz_r5tTmo",
            lipsyncModules: ["en"],
            cameraView: "upper"
        });

        await head.showAvatar({
            url: 'https://models.readyplayer.me/67e598cfce4c867ff24c9ea9.glb?morphTargets=ARKit,Oculus+Visemes,mouthOpen,mouthSmile,eyesClosed,eyesLookUp,eyesLookDown&textureSizeLimit=1024&textureFormat=png',
            body: 'F',
            avatarMood: 'neutral',
            ttsLang: "en-GB",
            ttsVoice: "en-GB-Standard-A",
            lipsyncLang: 'en'
        }, (ev) => {
            if (ev.lengthComputable) {
                let val = Math.min(100, Math.round(ev.loaded/ev.total * 100));
                nodeLoading.innerHTML = `<span>Loading ${val}%</span>`;
            }
        });
        nodeLoading.style.display = 'none';
    } catch (error) {
        console.error('Error initializing avatar:', error);
        nodeLoading.innerHTML = "<span>Error loading avatar</span>";
    }
}

// Set up event listeners
function setupEventListeners() {
    startButton.addEventListener('click', startInterview);
    sendButton.addEventListener('click', sendAnswer);
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault(); // Prevent newline in textarea
            sendAnswer();
        }
    });

    micButton.addEventListener('click', () => {
        if (isLiveChatActive) {
             console.log("Live chat is active, regular mic disabled.");
             return; // Do nothing if live chat is active
        }
        if (!isRecording) {
            startRecording();
        } else {
            stopRecording(); // Allow manual stop if needed
        }
    });

    // --- Add Live Chat Toggle Listener ---
    liveChatToggle.addEventListener('click', () => toggleLiveChat());
    // --- End Live Chat Toggle Listener ---

    closeOutfitAnalysis.addEventListener('click', () => {
        outfitAnalysisContainer.style.display = 'none';
        outfitAnalysisToggle.style.display = 'flex';
    });
    
    outfitAnalysisToggle.addEventListener('click', () => {
        if (outfitAnalysisData) {
            outfitAnalysisContainer.style.display = 'block';
            outfitAnalysisToggle.style.display = 'none';
        } else {
            // If no analysis data yet, capture a new frame
            captureAndSendFrame();
            statusIndicator.textContent = 'Analyzing your outfit...';
        }
    });
    
    closeExpressionAnalysis.addEventListener('click', () => {
        expressionAnalysisContainer.style.display = 'none';
        expressionAnalysisToggle.style.display = 'flex';
    });
    
    expressionAnalysisToggle.addEventListener('click', () => {
        expressionAnalysisContainer.style.display = 'block';
        expressionAnalysisToggle.style.display = 'none';
    });
    
    // Remove existing event listener if it exists
    endInterviewButton.removeEventListener('click', endInterview);
    
    // Add event listener with a named function for better reliability
    endInterviewButton.addEventListener('click', function endInterviewClickHandler() {
        // Deactivate live chat when ending interview
        if (isLiveChatActive) {
            toggleLiveChat(false);
        }
        endInterview(); // Call the original function
    });
    
    // Also handle the button by ID directly
    document.getElementById('endInterviewButton').onclick = function() {
        console.log('End Interview button clicked via direct onclick');
        endInterview();
    };
}

// Capture and send frame for outfit analysis
function captureAndSendFrame() {
    const canvas = document.createElement('canvas');
    const cameraElement = document.getElementById('cameraElement');
    canvas.width = cameraElement.videoWidth;
    canvas.height = cameraElement.videoHeight;
    
    if (canvas.width === 0 || canvas.height === 0) {
        console.error('Invalid canvas dimensions:', canvas.width, canvas.height);
        return;
    }
    
    const ctx = canvas.getContext('2d');
    ctx.drawImage(cameraElement, 0, 0, canvas.width, canvas.height);
    
    try {
        const imageData = canvas.toDataURL('image/jpeg', 0.8);
        if (!imageData || imageData.length < 22) { // Minimum length for a valid data URL
            console.error('Invalid image data generated');
            return;
        }
        
        const base64Data = imageData.split(',')[1];
        if (!base64Data) {
            console.error('Failed to extract base64 data');
            return;
        }
        
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({
                type: 'frame',
                data: base64Data
            }));
            console.log('Frame sent successfully');
        }
    } catch (error) {
        console.error('Error capturing frame:', error);
    }
}

// Start the interview process
function startInterview() {
    const topic = topicInput.value.trim();
    if (!topic) {
        alert('Please enter a topic for the interview');
        return;
    }

    startButton.disabled = true;
    topicInput.disabled = true;
    loadingElement.style.display = 'flex';
    outfitAnalysisToggle.style.display = 'flex';
    
    // Connect to WebSocket
    ws = new WebSocket('https://vm.tail9e6e2f.ts.net/ws');

    ws.onopen = () => {
        // Send the interview topic
        ws.send(JSON.stringify({
            type: 'start_interview',
            topic: topic
        }));
        
        // Start expression analysis
        toggleExpressionAnalysis(true);
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'question') {
            loadingElement.style.display = 'none';
            setupContainer.style.display = 'none';
            interviewContainer.style.display = 'block';
            addMessage(data.data);
        } else if (data.type === 'outfit_analysis') {
            formatOutfitAnalysis(data.data);
        } else if (data.type === 'expression_analysis') {
            // Handle incoming expression data from server
            updateExpressionAnalysis(data.data);
            expressionHistory.push(data.data);
        } else if (data.type === 'expression_summary') {
            // Handle server-generated expression summary
            formatExpressionSummary(data.data);
            
            // Hide loading indicator
            const reportLoadingIndicator = document.getElementById('reportLoadingIndicator');
            reportLoadingIndicator.style.display = 'none';
            
            // Display the report
            showExpressionReport();
        } else if (data.type === 'expression_status') {
            // Update status with server message
            statusIndicator.textContent = data.data;
        } else if (data.type === 'error') {
            loadingElement.style.display = 'none';
            addMessage(`Error: ${data.data}`);
            startButton.disabled = false;
            topicInput.disabled = false;
        } else if (data.type === 'combined_summary') {
            // Handle the combined expression and speech analysis summary
            formatExpressionSummary(data.data);
            
            // Hide loading indicator
            const reportLoadingIndicator = document.getElementById('reportLoadingIndicator');
            reportLoadingIndicator.style.display = 'none';
            
            // Display the report
            showExpressionReport();
        } else if (data.type === 'speech_analysis') {
            // Handle real-time speech analysis feedback
            console.log("Received speech analysis:", data.data);
            // You could add real-time feedback here if desired
        }
    };

    ws.onclose = () => {
        startButton.disabled = false;
        topicInput.disabled = false;
        loadingElement.style.display = 'none';
    };

    ws.onerror = (error) => {
        loadingElement.style.display = 'none';
        addMessage('Error connecting to the server. Please make sure the server is running.');
        startButton.disabled = false;
        topicInput.disabled = false;
    };
}

// Format and display outfit analysis
function formatOutfitAnalysis(text) {
    outfitAnalysisData = text;
    const sections = text.split('\n\n');
    const container = document.getElementById('outfitAnalysisContent');
    container.innerHTML = '';

    let rating = 0;
    sections.forEach(section => {
        if (section.trim()) {
            const sectionDiv = document.createElement('div');
            sectionDiv.className = 'outfit-analysis-section';

            if (section.includes('rating') || section.includes('Rating')) {
                // Extract the rating number from the text using a more precise pattern
                const ratingMatch = section.match(/Rating:\s*(\d+(?:\.\d+)?)\s*\/\s*10/i);
                if (ratingMatch) {
                    rating = parseFloat(ratingMatch[1]);
                    const fullStars = Math.floor(rating/2);
                    const halfStar = rating/2 - fullStars >= 0.5 ? 1 : 0;
                    const emptyStars = 5 - fullStars - halfStar;
                    
                    const starsHTML = 
                        '<i class="fas fa-star"></i>'.repeat(fullStars) + 
                        (halfStar ? '<i class="fas fa-star-half-alt"></i>' : '') +
                        '<i class="far fa-star"></i>'.repeat(emptyStars);
                    
                    sectionDiv.innerHTML = `
                        <div class="outfit-rating">
                            <div class="stars">${starsHTML}</div>
                            <div class="score">${rating}/10</div>
                        </div>
                    `;
                }
            } else if (section.includes('Overall Appropriateness')) {
                sectionDiv.innerHTML = `
                    <h3><i class="fas fa-check-circle"></i>Overall Appropriateness</h3>
                    <p>${section.split(':')[1].trim()}</p>
                `;
            } else if (section.includes('Professionalism')) {
                sectionDiv.innerHTML = `
                    <h3><i class="fas fa-user-tie"></i>Professionalism</h3>
                    <p>${section.split(':')[1].trim()}</p>
                `;
            } else if (section.includes('Color Coordination')) {
                sectionDiv.innerHTML = `
                    <h3><i class="fas fa-palette"></i>Color Coordination</h3>
                    <p>${section.split(':')[1].trim()}</p>
                `;
            } else if (section.includes('Suggestions')) {
                sectionDiv.innerHTML = `
                    <h3><i class="fas fa-lightbulb"></i>Suggestions</h3>
                    <p>${section.split(':')[1].trim()}</p>
                `;
            } else {
                sectionDiv.innerHTML = `<p>${section}</p>`;
            }

            container.appendChild(sectionDiv);
        }
    });

    outfitAnalysisContainer.style.display = 'block';
    outfitAnalysisToggle.style.display = 'none';
    statusIndicator.textContent = '';
}

// Add message to chat
function addMessage(message, isUser = false) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user-message' : 'ai-message'}`;
    messageDiv.textContent = message;
    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;

    if (!isUser) {
        speakMessage(message);
    }
}

// Speak the message using the avatar or fallback to browser speech synthesis
function speakMessage(message) {
    if (head) {
        // Stop any ongoing speech before starting new one
        head.stopSpeaking();
        head.speakText(message);
    } else {
        // Fallback to browser speech synthesis if avatar is not ready
        if (speechSynthesis.speaking) {
            speechSynthesis.cancel();
        }
        const utterance = new SpeechSynthesisUtterance(message);
        utterance.rate = 1;
        utterance.pitch = 1;
        speechSynthesis.speak(utterance);
    }
}

// Modified startRecording to handle regular mic only
function startRecording() {
    if (isRecording || isLiveChatActive) return; // Don't start if already recording or live chat active

    audioChunks = [];
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ audio: true })
            .then(stream => {
                // Use the existing 'recognition' instance for the button press
                if (!recognition) {
                    console.error("Regular speech recognition not initialized.");
                    return;
                }
                try {
                    isRecording = true;
                    micButton.classList.add('recording');
                    statusIndicator.textContent = 'Listening...';
                    userInput.placeholder = "Listening for your answer...";
                    recognition.start();
                    console.log("Regular Recognition started");
                } catch (e) {
                     console.error("Error starting regular recognition:", e);
                     statusIndicator.textContent = 'Mic error.';
                     isRecording = false;
                     micButton.classList.remove('recording');
                     userInput.placeholder = "Type your answer here...";
                }

                // MediaRecorder logic (if you still need raw audio chunks for something else)
                // If not, this part could be removed if only transcript is needed.
                mediaRecorder = new MediaRecorder(stream);
                mediaRecorder.ondataavailable = event => {
                    audioChunks.push(event.data);
                };
                mediaRecorder.start();

            })
            .catch(error => {
                console.error('Error accessing microphone:', error);
                statusIndicator.textContent = 'Mic access denied.';
                 userInput.placeholder = "Type your answer here...";
            });
    } else {
        console.error('getUserMedia not supported on your browser!');
        statusIndicator.textContent = 'Mic not supported.';
         userInput.placeholder = "Type your answer here...";
    }
}

// Modified stopRecording for regular mic button context
function stopRecording() {
    if (!isRecording) return; // Only stop if it was started via the button

    try {
        recognition.stop(); // Stop the regular recognition instance
        console.log("Regular Recognition stopped manually.");
    } catch (e) {
         console.error("Error stopping regular recognition:", e);
    }

    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
        // Stop the tracks to release the mic resource
        mediaRecorder.stream.getTracks().forEach(track => track.stop());
        mediaRecorder = null; // Clear recorder instance
    }
    isRecording = false;
    micButton.classList.remove('recording');
    statusIndicator.textContent = '';
    userInput.placeholder = "Type your answer here...";

    // Note: We don't send audio chunks here anymore,
    // as recognition.onresult handles sending the transcript for the regular mic.
}

// Send user's answer
function sendAnswer() {
    const answer = userInput.value.trim();
    if (!answer) return;

    addMessage(answer, true);
    userInput.value = '';
    statusIndicator.textContent = '';

    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
            type: 'answer',
            data: answer
        }));
    }
}

// Make endInterview globally available
window.endInterview = function() {
    console.log("Ending interview from global function...");
    
    // Show loading indicator
    const reportLoadingIndicator = document.getElementById('reportLoadingIndicator');
    reportLoadingIndicator.style.display = 'flex';
    
    // Disable button to prevent multiple clicks
    endInterviewButton.disabled = true;
    
    // Stop expression analysis
    if (isExpressionAnalysisActive) {
        isExpressionAnalysisActive = false;
        clearInterval(expressionInterval);
        expressionInterval = null;
        
        // Notify server to stop expression analysis
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({
                type: 'stop_expression_analysis'
            }));
        }
    }
    
    // Add a message about expression analysis
    addMessage("Your interview is complete! I'm analyzing your expressions throughout the interview. Your detailed report will be ready in a moment.", false);
    
    // Request expression summary from server
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
            type: 'get_expression_summary'
        }));
        
        // The server response will trigger the report display
        // See the ws.onmessage handler above
    } else {
        // Fallback if server connection is lost
        setTimeout(() => {
            reportLoadingIndicator.style.display = 'none';
            generateLocalExpressionSummary();
            showExpressionReport();
        }, 2500);
    }
    
    // Close WebSocket connection if open
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
            type: 'end_interview',
            data: 'Interview ended by user'
        }));
        
        setTimeout(() => {
            ws.close();
        }, 1000);
    }
    
    // Disable input fields
    userInput.disabled = true;
    sendButton.disabled = true;
    micButton.style.pointerEvents = 'none';
    
    // Update status
    statusIndicator.textContent = 'Interview ended. Preparing your expression report...';
};

// Generate local expression summary as fallback if server summary fails
function generateLocalExpressionSummary() {
    console.log("Generating local expression summary fallback...");
    
    // Check if we have any data to analyze
    if (expressionHistory.length === 0) {
        expressionSummary.innerHTML = `
            <h3>Interview Expression Analysis</h3>
            <p>No expression data was collected during this interview. Please ensure your camera is enabled and your face is visible during future interviews.</p>
        `;
        return;
    }
    
    // Use existing client-side summary generator
    generateExpressionSummary();
}

// Show expression report after receiving server summary
function showExpressionReport() {
    // Show expression summary panel
    expressionAnalysisContainer.style.display = 'block';
    expressionAnalysisToggle.style.display = 'none';
    
    // Highlight the report by scrolling to it
    expressionSummary.scrollIntoView({ behavior: 'smooth' });
    
    // Add a final message to guide user
    addMessage("Your expression report is now ready! You can review how your facial expressions changed during the interview. This feedback will help you prepare for future interviews.", false);
}

// Format the server-generated expression summary
function formatExpressionSummary(summaryData) {
    console.log("Formatting server expression summary:", summaryData);
    
    // Check if we have a combined summary with both expression and speech data
    const isExpressAndSpeechData = summaryData.expressionAnalysis && summaryData.speechAnalysis;
    
    // Extract the expression data (either from the combined structure or use the data as-is)
    const expressionData = isExpressAndSpeechData ? summaryData.expressionAnalysis : summaryData;
    
    // Extract speech data if available
    const speechData = isExpressAndSpeechData ? summaryData.speechAnalysis : null;
    
    if (!expressionData || (expressionData.summary && expressionData.summary.includes("No expression data"))) {
        expressionSummary.innerHTML = `
            <h3>Interview Expression Analysis</h3>
            <p>No expression data was collected during this interview. Please ensure your camera is enabled and your face is visible during future interviews.</p>
        `;
        
        // If we have speech data but no expression data, still show speech analysis
        if (speechData) {
            formatSpeechSummary(speechData);
        }
        
        return;
    }
    
    const averages = expressionData.averages || {};
    const trends = expressionData.trends || {};
    const engagement = expressionData.engagement || {};
    const recommendations = expressionData.recommendations || [];
    const recordCount = expressionData.recordCount || 0;
    
    // Get the text description of trends
    const smileTrend = trends.smileTrend || '';
    const lookAwayTrend = trends.lookAwayTrend || '';
    
    // Create engagement color based on level
    let engagementColor = '#FFA500'; // Orange (Average)
    if (engagement.level === 'Excellent') {
        engagementColor = '#008000'; // Green
    } else if (engagement.level === 'Good') {
        engagementColor = '#90EE90'; // Light green
    } else if (engagement.level === 'Poor') {
        engagementColor = '#FF0000'; // Red
    } else if (engagement.level === 'Needs Improvement') {
        engagementColor = '#FFA07A'; // Light salmon
    }
    
    // Create the HTML content for the summary
    let htmlContent = `
        <h3>Interview Expression Analysis</h3>
        <p>This analysis is based on ${recordCount} data points collected during your interview.</p>
        
        <div class="summary-section">
            <h4>Expression Averages</h4>
            <div class="expression-bars">
                <div class="expression-bar">
                    <div class="bar-label">Smile</div>
                    <div class="bar-container">
                        <div class="bar-fill" style="width: ${Math.min(100, averages.smile)}%; background-color: #4CAF50;"></div>
                    </div>
                    <div class="bar-value">${averages.smile}%</div>
                </div>
                <div class="expression-bar">
                    <div class="bar-label">Neutral</div>
                    <div class="bar-container">
                        <div class="bar-fill" style="width: ${Math.min(100, averages.neutral)}%; background-color: #2196F3;"></div>
                    </div>
                    <div class="bar-value">${averages.neutral}%</div>
                </div>
                <div class="expression-bar">
                    <div class="bar-label">Frown</div>
                    <div class="bar-container">
                        <div class="bar-fill" style="width: ${Math.min(100, averages.frown)}%; background-color: #FF9800;"></div>
                    </div>
                    <div class="bar-value">${averages.frown}%</div>
                </div>
                <div class="expression-bar">
                    <div class="bar-label">Looking Away</div>
                    <div class="bar-container">
                        <div class="bar-fill" style="width: ${Math.min(100, averages.lookAway)}%; background-color: #F44336;"></div>
                    </div>
                    <div class="bar-value">${averages.lookAway}%</div>
                </div>
                <div class="expression-bar">
                    <div class="bar-label">Blinking</div>
                    <div class="bar-container">
                        <div class="bar-fill" style="width: ${Math.min(100, averages.blink)}%; background-color: #9C27B0;"></div>
                    </div>
                    <div class="bar-value">${averages.blink}%</div>
                </div>
            </div>
        </div>
        
        <div class="summary-section">
            <h4>Expression Trends</h4>
            <p>Changes in your expressions from the beginning to the end of the interview:</p>
            <ul>
                <li>Smile: <span class="${trends.smileChange > 0 ? 'positive' : 'negative'}">${smileTrend}</span> 
                    <span class="trend-percent">(${trends.smileChange > 0 ? '+' : ''}${trends.smileChange}%)</span>
                </li>
                <li>Eye Contact: <span class="${trends.lookAwayChange < 0 ? 'positive' : 'negative'}">${lookAwayTrend}</span>
                    <span class="trend-percent">(${trends.lookAwayChange > 0 ? '+' : ''}${trends.lookAwayChange}% look away)</span>
                </li>
            </ul>
            <p class="trend-note">Note: For eye contact, negative percentages mean improvement (less looking away).</p>
        </div>
        
        <div class="summary-section">
            <h4>Overall Engagement Level</h4>
            <p>Your overall engagement score: <span style="color: ${engagementColor}; font-weight: bold;">${engagement.level} (${engagement.score})</span></p>
        </div>
        
        <div class="summary-section">
            <h4>Personalized Recommendations</h4>
            <ul>
                ${recommendations.map(rec => `<li>${rec}</li>`).join('')}
            </ul>
        </div>
    `;
    
    // Update the summary element
    expressionSummary.innerHTML = htmlContent;
    
    // Add styles for the new elements if they don't exist
    if (!document.getElementById('expression-summary-styles')) {
        const style = document.createElement('style');
        style.id = 'expression-summary-styles';
        style.textContent = `
            .summary-section {
                margin-bottom: 20px;
                background-color: #f5f5f5;
                padding: 15px;
                border-radius: 8px;
            }
            
            .expression-bars {
                margin-top: 10px;
            }
            
            .expression-bar {
                display: flex;
                align-items: center;
                margin-bottom: 8px;
            }
            
            .bar-label {
                width: 100px;
                font-weight: 500;
            }
            
            .bar-container {
                flex-grow: 1;
                height: 20px;
                background-color: #e0e0e0;
                border-radius: 4px;
                overflow: hidden;
            }
            
            .bar-fill {
                height: 100%;
                transition: width 1s ease-in-out;
            }
            
            .bar-value {
                width: 50px;
                text-align: right;
                margin-left: 10px;
                font-weight: 500;
            }
            
            .positive {
                color: green;
                font-weight: 500;
            }
            
            .negative {
                color: red;
                font-weight: 500;
            }
            
            .trend-percent {
                font-size: 0.9em;
                color: #555;
                margin-left: 5px;
            }
            
            .trend-note {
                font-size: 0.85em;
                font-style: italic;
                color: #666;
                margin-top: 10px;
            }
            
            /* Speech analysis styles */
            .filler-word {
                display: inline-block;
                padding: 5px 10px;
                margin: 5px;
                background-color: #f0f0f0;
                border-radius: 15px;
                font-size: 0.9em;
            }
            
            .filler-count {
                background-color: #e74c3c;
                color: white;
                border-radius: 50%;
                padding: 2px 6px;
                font-size: 0.8em;
                margin-left: 5px;
            }
            
            .fluency-score {
                font-size: 1.2em;
                font-weight: bold;
            }
            
            .speech-metrics {
                display: flex;
                flex-wrap: wrap;
                gap: 20px;
                margin: 15px 0;
            }
            
            .metric-box {
                background-color: #e8f4f8;
                border-radius: 8px;
                padding: 10px;
                min-width: 120px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            
            .metric-title {
                font-size: 0.85em;
                color: #555;
                margin-bottom: 5px;
            }
            
            .metric-value {
                font-size: 1.2em;
                font-weight: bold;
            }
        `;
        
        document.head.appendChild(style);
    }
    
    // If we have speech data, format and append it
    if (speechData) {
        formatSpeechSummary(speechData);
    }
}

// Format and display speech analysis summary
function formatSpeechSummary(speechData) {
    if (!speechData || (speechData.summary && speechData.summary.includes("Not enough speech data"))) {
        // Append to existing content or create new content
        const speechSection = document.createElement('div');
        speechSection.innerHTML = `
            <h3><i class="fas fa-comment-dots"></i> Verbal Communication Analysis</h3>
            <p>Not enough speech data was collected to provide analysis. Please provide more verbal responses during the interview.</p>
        `;
        expressionSummary.appendChild(speechSection);
        return;
    }
    
    const metrics = speechData.metrics || {};
    const fluency = speechData.fluency || {};
    const recommendations = speechData.recommendations || [];
    const mostCommonFillers = speechData.mostCommonFillers || [];
    const answerCount = speechData.answerCount || 0;
    
    // Extract additional metrics if available
    const structure = speechData.structure || {};
    const vocabulary = {
        unique: speechData.unique_words || 0,
        ratio: speechData.vocabulary_ratio || 0
    };
    
    // Get fluency color based on level
    let fluencyColor = '#FFA500'; // Orange (Average)
    if (fluency.level === 'Excellent' || fluency.level === 'Very Good') {
        fluencyColor = '#008000'; // Green
    } else if (fluency.level === 'Good') {
        fluencyColor = '#90EE90'; // Light green
    } else if (fluency.level === 'Poor') {
        fluencyColor = '#FF0000'; // Red
    } else if (fluency.level === 'Needs Improvement') {
        fluencyColor = '#FFA07A'; // Light salmon
    }
    
    // Create the speech summary HTML
    const speechSection = document.createElement('div');
    speechSection.innerHTML = `
        <h3><i class="fas fa-comment-dots"></i> Verbal Communication Analysis</h3>
        <p>This analysis is based on ${answerCount} answers provided during your interview.</p>
        
        <div class="summary-section">
            <div class="section-header">
                <h4><i class="fas fa-chart-bar"></i> Speech Metrics</h4>
                <button id="verbAnalysisButton" class="check-analysis-btn">
                    <i class="fas fa-external-link-alt"></i> View Detailed Report
                </button>
            </div>
            <div class="speech-metrics">
                <div class="metric-box primary-metric">
                    <div class="metric-title">Fluency Score</div>
                    <div class="metric-value" style="color: ${fluencyColor};">${metrics.avgFluency}</div>
                    <div class="metric-subtitle">${fluency.level}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-title">Total Words</div>
                    <div class="metric-value">${metrics.totalWords}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-title">Filler Words</div>
                    <div class="metric-value">${metrics.totalFillers}</div>
                    <div class="metric-subtitle">${metrics.fillerDensity}% of speech</div>
                </div>
                ${vocabulary.unique ? `
                <div class="metric-box">
                    <div class="metric-title">Vocabulary</div>
                    <div class="metric-value">${vocabulary.unique}</div>
                    <div class="metric-subtitle">Unique words</div>
                </div>` : ''}
                ${structure.sentences ? `
                <div class="metric-box">
                    <div class="metric-title">Sentence Length</div>
                    <div class="metric-value">${structure.words_per_sentence}</div>
                    <div class="metric-subtitle">Words per sentence</div>
                </div>` : ''}
            </div>
        </div>
        
        ${mostCommonFillers.length > 0 ? `
        <div class="summary-section">
            <h4><i class="fas fa-exclamation-circle"></i> Filler Word Analysis</h4>
            <p>These are verbal crutches that can make you sound less confident:</p>
            <div class="filler-words-container">
                ${mostCommonFillers.map(([word, count]) => 
                    `<span class="filler-word">${word} <span class="filler-count">${count}</span></span>`
                ).join('')}
            </div>
        </div>
        ` : ''}
        
        <div class="summary-section">
            <h4><i class="fas fa-exchange-alt"></i> Speech Patterns</h4>
            <div class="pattern-metrics">
                <div class="pattern-metric ${vocabulary.ratio > 40 ? 'positive' : vocabulary.ratio < 25 ? 'negative' : ''}">
                    <div class="pattern-label">Vocabulary Diversity</div>
                    <div class="pattern-value">${vocabulary.ratio}%</div>
                    <div class="pattern-description">
                        ${vocabulary.ratio > 40 ? 'Excellent variety of words' : 
                          vocabulary.ratio > 30 ? 'Good vocabulary range' : 
                          vocabulary.ratio > 20 ? 'Average word diversity' : 'Limited vocabulary range'}
                    </div>
                </div>
                
                <div class="pattern-metric ${structure.words_per_sentence > 30 || structure.words_per_sentence < 5 ? 'negative' : structure.words_per_sentence > 10 && structure.words_per_sentence < 20 ? 'positive' : ''}">
                    <div class="pattern-label">Sentence Structure</div>
                    <div class="pattern-value">${structure.sentences} sentences</div>
                    <div class="pattern-description">
                        ${structure.long_sentences > 0 ? `${structure.long_sentences} long, potentially rambling sentences` : ''}
                        ${structure.short_sentences > 2 ? `${structure.short_sentences} very short, potentially fragmented sentences` : ''}
                        ${structure.long_sentences === 0 && structure.short_sentences <= 2 ? 'Well-balanced sentence structure' : ''}
                    </div>
                </div>
                
                <div class="pattern-metric ${fluency.trend > 5 ? 'positive' : fluency.trend < -5 ? 'negative' : ''}">
                    <div class="pattern-label">Fluency Trend</div>
                    <div class="pattern-value">${fluency.trendDescription}</div>
                    <div class="pattern-description">
                        ${fluency.trend > 0 ? 'Your speaking fluency improved during the interview' : 
                          fluency.trend < 0 ? 'Your speaking became less fluent as the interview progressed' : 
                          'Your speaking fluency remained consistent'}
                    </div>
                </div>
            </div>
        </div>
        
        <div class="summary-section">
            <h4><i class="fas fa-bullseye"></i> Speech Improvement Recommendations</h4>
            <ul class="speech-recommendations">
                ${recommendations.map(rec => `<li>${rec}</li>`).join('')}
            </ul>
        </div>
    `;
    
    // Append to the expression summary element
    expressionSummary.appendChild(speechSection);
    
    // Add the event listener for the detailed report button
    setTimeout(() => {
        const verbAnalysisButton = document.getElementById('verbAnalysisButton');
        if (verbAnalysisButton) {
            verbAnalysisButton.addEventListener('click', () => {
                showDetailedVerbalAnalysis(speechData);
            });
        }
    }, 100);
}

// Function to show a detailed verbal analysis modal
function showDetailedVerbalAnalysis(speechData) {
    // Create modal container if it doesn't exist
    let modalContainer = document.getElementById('verbalAnalysisModal');
    if (!modalContainer) {
        modalContainer = document.createElement('div');
        modalContainer.id = 'verbalAnalysisModal';
        modalContainer.className = 'modal-container';
        document.body.appendChild(modalContainer);
        
        // Add styles for the modal
        const style = document.createElement('style');
        style.textContent = `
            .modal-container {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(0, 0, 0, 0.7);
                display: flex;
                justify-content: center;
                align-items: center;
                z-index: 1000;
            }
            
            .modal-content {
                background-color: white;
                border-radius: 8px;
                width: 80%;
                max-width: 800px;
                max-height: 90vh;
                overflow-y: auto;
                padding: 20px;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
            }
            
            .modal-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                border-bottom: 1px solid #eee;
                padding-bottom: 10px;
                margin-bottom: 20px;
            }
            
            .modal-close {
                background: none;
                border: none;
                font-size: 24px;
                cursor: pointer;
                color: #777;
            }
            
            .modal-close:hover {
                color: #333;
            }
            
            .verbal-section {
                margin-bottom: 25px;
            }
            
            .verbal-grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                gap: 15px;
                margin: 15px 0;
            }
            
            .verbal-item {
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 6px;
                border-left: 4px solid #4285f4;
            }
            
            .verbal-chart {
                width: 100%;
                height: 300px;
                margin: 20px 0;
                border: 1px solid #eee;
                border-radius: 6px;
            }
            
            .score-meter {
                width: 100%;
                height: 30px;
                background-color: #eee;
                border-radius: 15px;
                position: relative;
                overflow: hidden;
                margin: 15px 0;
            }
            
            .score-fill {
                height: 100%;
                border-radius: 15px;
                background: linear-gradient(90deg, #ff6b6b 0%, #ffb347 50%, #48d39a 100%);
                transition: width 1s ease-in-out;
            }
            
            .score-marker {
                position: absolute;
                width: 4px;
                height: 30px;
                background-color: #333;
                top: 0;
                transform: translateX(-50%);
                z-index: 1;
            }
            
            .score-labels {
                display: flex;
                justify-content: space-between;
                font-size: 12px;
                color: #666;
                margin-top: 5px;
            }
        `;
        document.head.appendChild(style);
    }
    
    // Extract insights from the speech data
    const metrics = speechData.metrics || {};
    const fluency = speechData.fluency || {};
    const structure = speechData.structure || {};
    const fillerWords = speechData.filler_words || {};
    const uniqueWords = speechData.unique_words || 0;
    const vocabularyRatio = speechData.vocabulary_ratio || 0;
    
    // Build array of filler words for display
    const fillerWordsArray = Object.entries(fillerWords).map(([word, count]) => ({ word, count }))
        .sort((a, b) => b.count - a.count);
    
    // Populate the modal
    modalContainer.innerHTML = `
        <div class="modal-content">
            <div class="modal-header">
                <h2><i class="fas fa-microphone-alt"></i> Detailed Verbal Communication Analysis</h2>
                <button class="modal-close">&times;</button>
            </div>
            
            <div class="verbal-section">
                <h3>Overall Fluency Assessment</h3>
                <p>Your speech fluency score is <strong>${metrics.avgFluency}/100</strong>, which is considered <strong>${fluency.level}</strong>.</p>
                
                <div class="score-meter">
                    <div class="score-fill" style="width: ${metrics.avgFluency}%;"></div>
                    <div class="score-marker" style="left: ${metrics.avgFluency}%;"></div>
                </div>
                <div class="score-labels">
                    <span>Poor (0)</span>
                    <span>Needs Improvement (40)</span>
                    <span>Average (60)</span>
                    <span>Good (80)</span>
                    <span>Excellent (100)</span>
                </div>
                
                <p>Throughout the interview, your fluency ${
                    fluency.trend > 10 ? 'significantly improved' : 
                    fluency.trend > 5 ? 'showed improvement' :
                    fluency.trend < -10 ? 'significantly decreased' :
                    fluency.trend < -5 ? 'slightly decreased' :
                    'remained consistent'
                } (${fluency.trend > 0 ? '+' : ''}${fluency.trend} points).</p>
            </div>
            
            <div class="verbal-section">
                <h3>Speech Metrics</h3>
                <div class="verbal-grid">
                    <div class="verbal-item">
                        <h4>Total Words</h4>
                        <p class="metric-large">${metrics.totalWords}</p>
                        <p class="metric-context">${
                            metrics.totalWords > 500 ? 'Very detailed responses' :
                            metrics.totalWords > 300 ? 'Detailed responses' :
                            metrics.totalWords > 150 ? 'Adequately detailed' :
                            metrics.totalWords > 50 ? 'Brief responses' : 'Very brief responses'
                        }</p>
                    </div>
                    
                    <div class="verbal-item">
                        <h4>Filler Words</h4>
                        <p class="metric-large">${metrics.totalFillers}</p>
                        <p class="metric-context">${metrics.fillerDensity}% of your speech</p>
                    </div>
                    
                    <div class="verbal-item">
                        <h4>Word-to-Filler Ratio</h4>
                        <p class="metric-large">${metrics.words_per_filler ? Math.round(metrics.words_per_filler) : 'N/A'}</p>
                        <p class="metric-context">Words per filler (higher is better)</p>
                    </div>
                    
                    <div class="verbal-item">
                        <h4>Vocabulary Diversity</h4>
                        <p class="metric-large">${vocabularyRatio}%</p>
                        <p class="metric-context">${uniqueWords} unique words used</p>
                    </div>
                </div>
            </div>
            
            <div class="verbal-section">
                <h3>Sentence Structure</h3>
                <div class="verbal-grid">
                    <div class="verbal-item">
                        <h4>Sentences</h4>
                        <p class="metric-large">${structure.sentences || 'N/A'}</p>
                    </div>
                    
                    <div class="verbal-item">
                        <h4>Words Per Sentence</h4>
                        <p class="metric-large">${structure.words_per_sentence || 'N/A'}</p>
                        <p class="metric-context">${
                            structure.words_per_sentence > 25 ? 'Your sentences tend to be long' :
                            structure.words_per_sentence < 8 ? 'Your sentences tend to be short' :
                            'Your sentences have a good length'
                        }</p>
                    </div>
                    
                    <div class="verbal-item">
                        <h4>Long Sentences</h4>
                        <p class="metric-large">${structure.long_sentences || 0}</p>
                        <p class="metric-context">Sentences > 25 words</p>
                    </div>
                    
                    <div class="verbal-item">
                        <h4>Short Fragments</h4>
                        <p class="metric-large">${structure.short_sentences || 0}</p>
                        <p class="metric-context">Sentences < 3 words</p>
                    </div>
                </div>
            </div>
            
            ${fillerWordsArray.length > 0 ? `
            <div class="verbal-section">
                <h3>Filler Word Analysis</h3>
                <p>Filler words can make you sound less confident and prepared. Here are your most common verbal crutches:</p>
                <div class="verbal-grid">
                    ${fillerWordsArray.slice(0, 8).map(item => `
                        <div class="verbal-item">
                            <h4>"${item.word}"</h4>
                            <p class="metric-large">${item.count}</p>
                            <p class="metric-context">Used ${item.count} times</p>
                        </div>
                    `).join('')}
                </div>
                
                <div class="verbal-tip">
                    <h4><i class="fas fa-lightbulb"></i> How to Reduce Filler Words</h4>
                    <ul>
                        <li>Practice replacing fillers with deliberate pauses</li>
                        <li>Record yourself speaking and identify patterns</li>
                        <li>Slow down your speech to give yourself time to think</li>
                        <li>Prepare talking points in advance for important conversations</li>
                    </ul>
                </div>
            </div>
            ` : ''}
            
            <div class="verbal-section">
                <h3>Key Takeaways</h3>
                <ul>
                    <li>Your overall verbal fluency is <strong>${fluency.level}</strong> with a score of <strong>${metrics.avgFluency}/100</strong></li>
                    <li>You used <strong>${metrics.totalFillers}</strong> filler words (${metrics.fillerDensity}% of your speech)</li>
                    <li>Your vocabulary diversity is <strong>${vocabularyRatio}%</strong> (${
                        vocabularyRatio > 40 ? 'excellent' :
                        vocabularyRatio > 30 ? 'good' :
                        vocabularyRatio > 20 ? 'average' : 'below average'
                    })</li>
                    <li>Your average sentence length is <strong>${structure.words_per_sentence || 'N/A'}</strong> words</li>
                </ul>
            </div>
            
            <div class="verbal-section">
                <button class="close-detail-btn">Close Report</button>
            </div>
        </div>
    `;
    
    // Add event listeners to close the modal
    const closeButtons = modalContainer.querySelectorAll('.modal-close, .close-detail-btn');
    closeButtons.forEach(button => {
        button.addEventListener('click', () => {
            modalContainer.style.display = 'none';
        });
    });
    
    // Click outside to close
    modalContainer.addEventListener('click', (e) => {
        if (e.target === modalContainer) {
            modalContainer.style.display = 'none';
        }
    });
    
    // Make the modal visible
    modalContainer.style.display = 'flex';
}
