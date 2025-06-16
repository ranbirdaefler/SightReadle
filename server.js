const express = require('express');
const path = require('path');
const fs = require('fs');
const app = express();
const PORT = process.env.PORT || 3000;

// Serve static files (CSS, JS, images, and MusicXML files)
app.use(express.static('public'));
app.use('/musicxml', express.static('musicxml')); // Serve MusicXML files

// Get list of available exercises (only those with .mxl files)
function getAvailableExercises() {
    const musicxmlDir = path.join(__dirname, 'musicxml');
    const availableExercises = [];
    
    try {
        const files = fs.readdirSync(musicxmlDir);
        
        files.forEach(file => {
            if (file.endsWith('.mxl')) {
                const match = file.match(/exercise_(\d+)\.mxl/);
                if (match) {
                    const exerciseNumber = parseInt(match[1]);
                    availableExercises.push(exerciseNumber);
                }
            }
        });
        
        availableExercises.sort((a, b) => a - b); // Sort numerically
        console.log(`Found ${availableExercises.length} exercises with .mxl files:`, availableExercises.slice(0, 10), '...');
        
    } catch (error) {
        console.error('Error reading musicxml directory:', error);
    }
    
    return availableExercises;
}

// Cache the available exercises list
const AVAILABLE_EXERCISES = getAvailableExercises();

// Function to get today's piece number (only from available exercises)
function getTodaysPieceNumber() {
    if (AVAILABLE_EXERCISES.length === 0) {
        return 1; // Fallback
    }
    
    const startDate = new Date('2025-01-01'); // Your start date
    const today = new Date();
    const diffTime = today - startDate;
    const diffDays = Math.floor(diffTime / (1000 * 60 * 60 * 24));
    const index = diffDays % AVAILABLE_EXERCISES.length;
    return AVAILABLE_EXERCISES[index];
}

// Function to get a random available exercise
function getRandomExercise() {
    if (AVAILABLE_EXERCISES.length === 0) {
        return 1; // Fallback
    }
    const randomIndex = Math.floor(Math.random() * AVAILABLE_EXERCISES.length);
    return AVAILABLE_EXERCISES[randomIndex];
}

// Function to get next/previous available exercise
function getNextExercise(currentExercise) {
    const currentIndex = AVAILABLE_EXERCISES.indexOf(currentExercise);
    if (currentIndex === -1) {
        return AVAILABLE_EXERCISES[0] || 1;
    }
    const nextIndex = (currentIndex + 1) % AVAILABLE_EXERCISES.length;
    return AVAILABLE_EXERCISES[nextIndex];
}

function getPreviousExercise(currentExercise) {
    const currentIndex = AVAILABLE_EXERCISES.indexOf(currentExercise);
    if (currentIndex === -1) {
        return AVAILABLE_EXERCISES[0] || 1;
    }
    const prevIndex = currentIndex === 0 ? AVAILABLE_EXERCISES.length - 1 : currentIndex - 1;
    return AVAILABLE_EXERCISES[prevIndex];
}

// Route to get today's piece info
app.get('/api/today', (req, res) => {
    const pieceNumber = getTodaysPieceNumber();
    res.json({
        pieceNumber: pieceNumber,
        imageUrl: `/images/exercise_${pieceNumber}.png`,
        title: `Daily Sight Reading Challenge #${pieceNumber}`,
        date: new Date().toDateString(),
        hasScoring: true
    });
});

// Route to get specific piece (only if it has .mxl file)
app.get('/api/piece/:number', (req, res) => {
    const pieceNumber = parseInt(req.params.number);
    
    if (!AVAILABLE_EXERCISES.includes(pieceNumber)) {
        return res.status(404).json({ 
            error: `Exercise #${pieceNumber} not available (no .mxl file found)`,
            availableExercises: AVAILABLE_EXERCISES.slice(0, 20) // Show first 20 as example
        });
    }
    
    res.json({
        pieceNumber: pieceNumber,
        imageUrl: `/images/exercise_${pieceNumber}.png`,
        title: `Sight Reading Challenge #${pieceNumber}`,
        date: new Date().toDateString(),
        hasScoring: true
    });
});

// Route to get random piece
app.get('/api/random', (req, res) => {
    const pieceNumber = getRandomExercise();
    res.json({
        pieceNumber: pieceNumber,
        imageUrl: `/images/exercise_${pieceNumber}.png`,
        title: `Random Sight Reading Challenge #${pieceNumber}`,
        date: new Date().toDateString(),
        hasScoring: true
    });
});

// Route to get next piece
app.get('/api/next/:current', (req, res) => {
    const currentPiece = parseInt(req.params.current);
    const nextPiece = getNextExercise(currentPiece);
    res.json({
        pieceNumber: nextPiece,
        imageUrl: `/images/exercise_${nextPiece}.png`,
        title: `Sight Reading Challenge #${nextPiece}`,
        date: new Date().toDateString(),
        hasScoring: true
    });
});

// Route to get previous piece
app.get('/api/previous/:current', (req, res) => {
    const currentPiece = parseInt(req.params.current);
    const prevPiece = getPreviousExercise(currentPiece);
    res.json({
        pieceNumber: prevPiece,
        imageUrl: `/images/exercise_${prevPiece}.png`,
        title: `Sight Reading Challenge #${prevPiece}`,
        date: new Date().toDateString(),
        hasScoring: true
    });
});

// Route to get list of available exercises
app.get('/api/available', (req, res) => {
    res.json({
        exercises: AVAILABLE_EXERCISES,
        count: AVAILABLE_EXERCISES.length,
        range: AVAILABLE_EXERCISES.length > 0 ? 
            `${Math.min(...AVAILABLE_EXERCISES)} - ${Math.max(...AVAILABLE_EXERCISES)}` : 
            'None'
    });
});

// Serve the main page
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.listen(PORT, () => {
    console.log(`SightReadle server running on http://localhost:${PORT}`);
    console.log(`Available exercises: ${AVAILABLE_EXERCISES.length} (with .mxl files)`);
});
