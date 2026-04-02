const express = require('express');
const path = require('path');
const fs = require('fs');
const multer = require('multer');
const FormData = require('form-data');
const fetch = require('node-fetch');

const app = express();
const PORT = process.env.PORT || 3000;
const SCORING_SERVICE_URL = process.env.SCORING_URL || 'http://localhost:8001';
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 50 * 1024 * 1024 } });

app.use(express.json({ limit: '50mb' }));
app.use(express.static('public'));
app.use('/audio/segments', express.static('public/audio/segments'));
app.use('/musicxml/segments', express.static(path.join(__dirname, 'scmpa', 'data', 'segments', 'musicxml')));

// Load segment manifest
let SEGMENTS = [];
try {
    const manifestPath = path.join(__dirname, 'scmpa', 'data', 'segments', 'manifest.json');
    if (fs.existsSync(manifestPath)) {
        const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf-8'));
        SEGMENTS = manifest.segments || [];
        console.log(`Loaded ${SEGMENTS.length} segments from manifest`);
    }
} catch (error) {
    console.error('Error loading segment manifest:', error.message);
}

function formatSegment(segment) {
    return {
        segment_id: segment.id,
        source_piece: segment.source_piece,
        n_bars: segment.n_bars,
        n_notes: segment.n_notes,
        tempo: segment.tempo,
        time_signature: segment.time_signature,
        key_signature: segment.key_signature,
        duration_sec: segment.duration_sec,
        musicxml_url: `/musicxml/segments/${segment.id}.musicxml`,
        audio_url: `/audio/segments/${segment.id}.mp3`,
    };
}

// ── Daily Challenge ──
app.get('/api/today', (req, res) => {
    if (SEGMENTS.length === 0) {
        return res.status(503).json({ error: 'No segments loaded' });
    }
    const daysSinceEpoch = Math.floor(Date.now() / 86400000);
    const index = daysSinceEpoch % SEGMENTS.length;
    res.json({ ...formatSegment(SEGMENTS[index]), mode: 'daily', challenge_number: daysSinceEpoch });
});

// ── Random Practice ──
app.get('/api/random', (req, res) => {
    if (SEGMENTS.length === 0) {
        return res.status(503).json({ error: 'No segments loaded' });
    }
    const index = Math.floor(Math.random() * SEGMENTS.length);
    res.json({ ...formatSegment(SEGMENTS[index]), mode: 'random' });
});

// ── Stats ──
app.get('/api/stats', (req, res) => {
    res.json({
        total_segments: SEGMENTS.length,
        total_pieces: [...new Set(SEGMENTS.map(s => s.source_piece))].length,
    });
});

// ── Scoring proxy ──
app.post('/api/score', upload.single('audio'), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ error: 'No audio file provided' });
        }
        const form = new FormData();
        form.append('segment_id', req.body.segment_id);
        form.append('audio', req.file.buffer, { filename: 'recording.wav', contentType: 'audio/wav' });

        const response = await fetch(`${SCORING_SERVICE_URL}/score`, { method: 'POST', body: form });
        if (!response.ok) {
            const error = await response.text();
            console.error('Scoring service error:', error);
            return res.status(500).json({ error: 'Scoring failed' });
        }
        const scores = await response.json();
        res.json(scores);
    } catch (err) {
        console.error('Scoring error:', err.message);
        res.status(503).json({ error: 'Scoring service unavailable' });
    }
});

// ── Tempo re-render proxy ──
app.post('/api/render-tempo', upload.none(), async (req, res) => {
    try {
        const form = new FormData();
        form.append('segment_id', req.body.segment_id);
        form.append('bpm', req.body.bpm.toString());

        const response = await fetch(`${SCORING_SERVICE_URL}/render-tempo`, {
            method: 'POST', body: form, headers: form.getHeaders(),
        });
        if (!response.ok) {
            const errText = await response.text();
            console.error('Tempo render error:', errText);
            return res.status(500).json({ error: 'Tempo rendering failed' });
        }
        const arrayBuffer = await response.arrayBuffer();
        res.set('Content-Type', 'audio/mpeg');
        res.send(Buffer.from(arrayBuffer));
    } catch (err) {
        console.error('Tempo render error:', err.message);
        res.status(500).json({ error: 'Tempo rendering service unavailable' });
    }
});

// ── Seeded Leaderboard — flip to false when you have 50+ real daily players ──
const SEED_FAKE_SCORES = true;

function generateFakeScores(dateString) {
    let seed = 0;
    for (let i = 0; i < dateString.length; i++) {
        seed = ((seed << 5) - seed) + dateString.charCodeAt(i);
        seed = seed & seed;
    }

    function rand() {
        seed = (seed * 1664525 + 1013904223) & 0xFFFFFFFF;
        return (seed >>> 0) / 0xFFFFFFFF;
    }

    const dailyMean = 0.55 + rand() * 0.17;
    const stdDev = 0.14;
    const nPlayers = 35 + Math.floor(rand() * 40);

    const scores = [];
    for (let i = 0; i < nPlayers; i++) {
        const u1 = Math.max(rand(), 0.0001);
        const u2 = rand();
        const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);

        let s = dailyMean + z * stdDev;
        s = Math.max(0.15, Math.min(0.98, s));
        s = Math.round(s * 1000) / 1000;
        scores.push(s);
    }

    return scores;
}

// ── Daily Leaderboard ──
const SCORES_DIR = path.join(__dirname, 'data', 'daily_scores');
if (!fs.existsSync(SCORES_DIR)) {
    fs.mkdirSync(SCORES_DIR, { recursive: true });
}

function loadDayScores(date) {
    const filePath = path.join(SCORES_DIR, `${date}.json`);
    if (fs.existsSync(filePath)) {
        return JSON.parse(fs.readFileSync(filePath, 'utf8'));
    }
    return [];
}

function saveDayScores(date, scores) {
    fs.writeFileSync(path.join(SCORES_DIR, `${date}.json`), JSON.stringify(scores, null, 2));
}

function computeDistribution(scores) {
    const buckets = Array(10).fill(0);
    for (const s of scores) {
        buckets[Math.min(Math.floor(s * 10), 9)]++;
    }
    return buckets.map((count, i) => ({ range: `${i * 10}-${(i + 1) * 10}%`, count }));
}

app.post('/api/daily-score', express.json(), (req, res) => {
    const { user_id, score, n_correct, n_missed, n_extra, total_expected } = req.body;
    if (!user_id || score === undefined) {
        return res.status(400).json({ error: 'Missing user_id or score' });
    }

    const today = new Date().toISOString().split('T')[0];
    const entry = {
        date: today, user_id,
        score: Math.round(score * 1000) / 1000,
        n_correct: n_correct || 0, n_missed: n_missed || 0,
        n_extra: n_extra || 0, total_expected: total_expected || 0,
        timestamp: new Date().toISOString(),
    };

    let dayScores = loadDayScores(today);
    const existingIdx = dayScores.findIndex(s => s.user_id === user_id);
    if (existingIdx >= 0) {
        if (score > dayScores[existingIdx].score) dayScores[existingIdx] = entry;
    } else {
        dayScores.push(entry);
    }
    saveDayScores(today, dayScores);

    const realScores = dayScores.map(s => s.score);
    const allScores = SEED_FAKE_SCORES
        ? realScores.concat(generateFakeScores(today)).sort((a, b) => a - b)
        : realScores.sort((a, b) => a - b);

    const userScore = dayScores.find(s => s.user_id === user_id).score;
    const rank = allScores.filter(s => s > userScore).length + 1;
    const percentile = Math.round(
        (allScores.filter(s => s <= userScore).length / allScores.length) * 100
    );

    res.json({
        your_score: userScore, rank,
        total_players: allScores.length, percentile,
        top_score: allScores[allScores.length - 1],
        median_score: allScores[Math.floor(allScores.length / 2)],
        distribution: computeDistribution(allScores),
    });
});

// ── Daily Leaderboard (read-only, for histogram before playing) ──
app.get('/api/daily-leaderboard', (req, res) => {
    const today = new Date().toISOString().split('T')[0];
    const dayScores = loadDayScores(today);

    const realScores = dayScores.map(s => s.score);
    const allScores = SEED_FAKE_SCORES
        ? realScores.concat(generateFakeScores(today)).sort((a, b) => a - b)
        : realScores.sort((a, b) => a - b);

    res.json({
        total_players: allScores.length,
        distribution: computeDistribution(allScores),
        median_score: allScores.length ? allScores[Math.floor(allScores.length / 2)] : 0,
        top_score: allScores.length ? allScores[allScores.length - 1] : 0,
    });
});

// ── Scoring health ──
app.get('/api/scoring/health', async (req, res) => {
    try {
        const response = await fetch(`${SCORING_SERVICE_URL}/health`);
        const data = await response.json();
        res.json(data);
    } catch (err) {
        res.json({ status: 'unavailable', error: err.message });
    }
});

// DEBUG — REMOVE AFTER FIXING DEPLOYMENT
app.get('/debug/files', (req, res) => {
    const results = {
        cwd: process.cwd(),
        dirname: __dirname,
    };

    const checks = [
        'data',
        'data/segments',
        'data/segments/musicxml',
        'data/segments/midi',
        'data/segments/manifest.json',
        'public',
        'public/audio',
        'public/audio/segments',
        'server',
        'src',
    ];

    results.paths = {};
    for (const p of checks) {
        const fullPath = path.join(__dirname, p);
        const entry = {
            exists: fs.existsSync(fullPath),
            fullPath: fullPath,
        };

        if (entry.exists) {
            const stat = fs.statSync(fullPath);
            entry.isDirectory = stat.isDirectory();
            entry.isFile = stat.isFile();

            if (stat.isDirectory()) {
                const files = fs.readdirSync(fullPath);
                entry.totalFiles = files.length;
                entry.firstFiles = files.slice(0, 10);
            }

            if (stat.isFile()) {
                entry.sizeBytes = stat.size;
            }
        }

        results.paths[p] = entry;
    }

    try {
        results.rootContents = fs.readdirSync(__dirname);
    } catch (e) {
        results.rootContents = `Error: ${e.message}`;
    }

    results.note = "If musicxml directory exists but 404s, the express.static path is wrong.";

    res.json(results);
});
// END DEBUG

app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.listen(PORT, '0.0.0.0', () => {
    console.log(`SightReadle server running on port ${PORT}`);
    console.log(`Segments loaded: ${SEGMENTS.length}`);
    console.log(`Scoring service expected at: ${SCORING_SERVICE_URL}`);
});
