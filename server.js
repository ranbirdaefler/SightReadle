
const express = require('express');
const path = require('path');
const app = express();
const PORT = process.env.PORT || 3000;

// Serve static files (CSS, JS, images, and MusicXML files)
app.use(express.static('public'));
app.use('/musicxml', express.static('musicxml')); // Serve MusicXML files

// Function to get today's piece number (1-354)
function getTodaysPieceNumber() {
    const startDate = new Date('2025-01-01'); // Your start date
    const today = new Date();
    const diffTime = today - startDate;
    const diffDays = Math.floor(diffTime / (1000 * 60 * 60 * 24));
    return (diffDays % 354) + 1; // Cycle through 1-354
}

// Route to get today's piece info
app.get('/api/today', (req, res) => {
    const pieceNumber = getTodaysPieceNumber();
    res.json({
        pieceNumber: pieceNumber,
        imageUrl: `/images/exercise_${pieceNumber}.png`,
        title: `Daily Sight Reading Challenge #${pieceNumber}`,
        date: new Date().toDateString()
    });
});

// Route to get specific piece (for testing)
app.get('/api/piece/:number', (req, res) => {
    const pieceNumber = parseInt(req.params.number);
    if (pieceNumber < 1 || pieceNumber > 354) {
        return res.status(404).json({ error: 'Piece not found' });
    }
    
    res.json({
        pieceNumber: pieceNumber,
        imageUrl: `/images/exercise_${pieceNumber}.png`,
        title: `Sight Reading Challenge #${pieceNumber}`,
        date: new Date().toDateString()
    });
});

// Serve the main page
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.listen(PORT, () => {
    console.log(`SightReadle server running on http://localhost:${PORT}`);
});
