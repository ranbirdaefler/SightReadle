class SightReadle {
    constructor() {
        this.recorder = new AudioRecorder();
        this.currentSegment = null;
        this.currentMode = 'daily';
        this.recordingTimer = null;
        this.recordingStartTime = null;
        this.isRecording = false;
        this.osmd = null;

        this._originalTempo = 120;
        this._selectedTempo = 120;
        this._tempoDebounce = null;
        this._isLoadingTempo = false;

        this._metronomeOn = false;
        this._metronomeInterval = null;
        this._metronomeCtx = null;

        this._currentDifficulty = 'intermediate';
        this._currentBars = 4;

        this._bindModeButtons();
        this._bindControls();
        this._bindDifficultyButtons();
        this._bindBarButtons();
        this.loadDaily();
    }

    // ── Mode switching ──

    _bindModeButtons() {
        document.getElementById('daily-btn').addEventListener('click', () => this.loadDaily());
        document.getElementById('random-btn').addEventListener('click', () => this.showDifficultySelect());
        document.getElementById('refresh-btn').addEventListener('click', () => this.refreshRandom());
        document.getElementById('reconfig-btn').addEventListener('click', () => this.showDifficultySelect());
    }

    _bindControls() {
        document.getElementById('listen-btn').addEventListener('click', () => this._toggleReference());
        document.getElementById('record-btn').addEventListener('click', () => this.startRecording());
        document.getElementById('stop-btn').addEventListener('click', () => this.stopRecording());
        document.getElementById('try-again-btn').addEventListener('click', () => this.tryAgain());
        document.getElementById('tempo-slider').addEventListener('input', (e) => this._onTempoSlide(e));
        document.getElementById('tempo-reset-btn').addEventListener('click', () => this._resetTempo());
        document.getElementById('metronome-btn').addEventListener('click', () => this._toggleMetronome());
    }

    _bindDifficultyButtons() {
        document.querySelectorAll('.diff-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const diff = btn.dataset.diff;
                this._currentDifficulty = diff;
                document.querySelectorAll('.diff-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                this.showBarSelect();
            });
        });
    }

    _bindBarButtons() {
        document.querySelectorAll('.bar-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const bars = parseInt(btn.dataset.bars, 10);
                this._currentBars = bars;
                document.querySelectorAll('.bar-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                this.loadRandomSegment(bars);
            });
        });
    }

    // ── Daily ──

    async loadDaily() {
        this.currentMode = 'daily';
        this._dailyUserScore = null;
        document.getElementById('daily-btn').classList.add('active');
        document.getElementById('random-btn').classList.remove('active');
        document.getElementById('refresh-btn').style.display = 'none';
        document.getElementById('reconfig-btn').style.display = 'none';
        document.getElementById('difficulty-selector').classList.add('hidden');
        document.getElementById('bar-selector').classList.add('hidden');
        document.getElementById('challenge-card').style.display = '';

        try {
            const res = await fetch('/api/today');
            const data = await res.json();
            document.getElementById('challenge-title').textContent =
                `Daily Challenge #${data.challenge_number}`;
            this.displaySegment(data);

            const saved = this._getSavedDailyResult(data.challenge_number);
            if (saved) {
                this._restoreSavedResult(saved);
                this._dailyUserScore = saved.userScore;
            }

            this._loadDailyHistogram();
        } catch (err) {
            this._showError('Failed to load daily challenge');
        }
    }

    // ── Random practice flow ──

    showDifficultySelect() {
        this.currentMode = 'random';
        document.getElementById('daily-btn').classList.remove('active');
        document.getElementById('random-btn').classList.add('active');
        document.getElementById('difficulty-selector').classList.remove('hidden');
        document.getElementById('bar-selector').classList.add('hidden');
        document.getElementById('challenge-card').style.display = 'none';
        document.getElementById('leaderboard-section').classList.add('hidden');
        document.querySelectorAll('.diff-btn').forEach(b => b.classList.remove('active'));
    }

    showBarSelect() {
        document.getElementById('bar-selector').classList.remove('hidden');
        document.querySelectorAll('.bar-btn').forEach(b => b.classList.remove('active'));
    }

    async loadRandomSegment(bars) {
        this._currentBars = bars;
        this._dailyUserScore = null;

        document.getElementById('difficulty-selector').classList.add('hidden');
        document.getElementById('bar-selector').classList.add('hidden');
        document.getElementById('challenge-card').style.display = '';
        document.getElementById('reconfig-btn').style.display = 'inline-block';
        document.getElementById('refresh-btn').style.display = 'inline-block';
        document.getElementById('leaderboard-section').classList.add('hidden');

        const titleParts = [
            'Random Practice',
            this._currentDifficulty.charAt(0).toUpperCase() + this._currentDifficulty.slice(1),
            `${bars} bars`,
        ];
        document.getElementById('challenge-title').textContent = titleParts.join(' \u2014 ');

        try {
            const recentPieces = encodeURIComponent(JSON.stringify(this._getRecentPieces()));
            const recentSegs = encodeURIComponent(JSON.stringify(this._getRecentSegments()));
            const url = `/api/random?difficulty=${this._currentDifficulty}&bars=${bars}&recent_pieces=${recentPieces}&recent_segs=${recentSegs}`;
            const res = await fetch(url);
            if (!res.ok) {
                const err = await res.json().catch(() => ({}));
                throw new Error(err.error || 'Failed to get segment');
            }
            const data = await res.json();
            if (data.source_piece) this._addRecentPiece(data.source_piece);
            if (data.segment_id || data.id) this._addRecentSegment(data.segment_id || data.id);
            this.displaySegment(data);
        } catch (err) {
            this._showError(err.message || 'Failed to load segment');
        }
    }

    async refreshRandom() {
        await this.loadRandomSegment(this._currentBars);
    }

    // ── Recent tracking (localStorage) ──

    _getRecentPieces() {
        try { return JSON.parse(localStorage.getItem('sightreadle_recent_pieces') || '[]'); }
        catch { return []; }
    }

    _addRecentPiece(sourcePiece) {
        let recent = this._getRecentPieces();
        if (!recent.includes(sourcePiece)) recent.push(sourcePiece);
        if (recent.length > 10) recent = [];
        localStorage.setItem('sightreadle_recent_pieces', JSON.stringify(recent));
    }

    _getRecentSegments() {
        try { return JSON.parse(localStorage.getItem('sightreadle_recent_segments') || '[]'); }
        catch { return []; }
    }

    _addRecentSegment(segmentId) {
        let recent = this._getRecentSegments();
        if (!recent.includes(segmentId)) recent.push(segmentId);
        if (recent.length > 50) recent = recent.slice(-30);
        localStorage.setItem('sightreadle_recent_segments', JSON.stringify(recent));
    }

    // ── Segment display ──

    displaySegment(segment) {
        this.currentSegment = segment;

        const pieceName = this._formatPieceName(segment.source_piece);
        const timeSig = segment.time_signature
            ? `${segment.time_signature[0]}/${segment.time_signature[1]}` : '4/4';
        const keySig = segment.key_signature || '';
        const diff = segment.difficulty
            ? segment.difficulty.charAt(0).toUpperCase() + segment.difficulty.slice(1) : '';

        let metaText = `${pieceName} \u00B7 ${segment.n_bars} bars \u00B7 ${segment.n_notes} notes \u00B7 ${timeSig}`;
        if (keySig) metaText += ` \u00B7 ${keySig}`;
        if (diff) metaText += ` \u00B7 ${diff}`;
        document.getElementById('challenge-meta').textContent = metaText;

        document.getElementById('results-section').classList.add('hidden');
        document.getElementById('record-btn').classList.remove('hidden');

        this._initTempoSlider(segment.tempo);

        if (segment.musicxml_url) {
            this._renderSheetMusic(segment.musicxml_url);
        }

        const segId = segment.segment_id || segment.id;
        this._loadReferenceAtTempo(segId, this._selectedTempo);
    }

    async _renderSheetMusic(musicxmlUrl) {
        const container = document.getElementById('sheet-music-container');
        try {
            const response = await fetch(musicxmlUrl);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const xmlText = await response.text();

            await new Promise(resolve => requestAnimationFrame(resolve));

            container.innerHTML = '';
            this.osmd = new opensheetmusicdisplay.OpenSheetMusicDisplay(
                container,
                {
                    autoResize: true,
                    backend: 'svg',
                    drawTitle: false,
                    drawSubtitle: false,
                    drawComposer: false,
                    drawLyricist: false,
                    drawPartNames: false,
                    drawPartAbbreviations: false,
                    drawMeasureNumbers: false,
                    drawCredits: false,
                    drawTimeSignatures: true,
                    drawKeySignatures: true,
                    drawMetronomeMarks: false,
                }
            );

            await this.osmd.load(xmlText);
            this.osmd.render();
        } catch (err) {
            console.error('Sheet music rendering failed:', err);
            container.innerHTML = '<div class="loading" style="color:#e53e3e">Failed to load sheet music</div>';
        }
    }

    // ── Tempo control ──

    _initTempoSlider(originalTempo) {
        this._originalTempo = originalTempo || 120;
        const defaultBpm = Math.round(this._originalTempo * 0.7 / 5) * 5;

        const slider = document.getElementById('tempo-slider');
        slider.min = Math.max(30, Math.round(this._originalTempo * 0.3));
        slider.max = Math.round(this._originalTempo * 1.2);
        slider.value = defaultBpm;
        this._selectedTempo = defaultBpm;
        this._updateTempoDisplay(defaultBpm);
    }

    _updateTempoDisplay(bpm) {
        const el = document.getElementById('tempo-display');
        const pct = Math.round((bpm / this._originalTempo) * 100);
        el.textContent = `\u2669 = ${bpm} (${pct}%)`;
    }

    _onTempoSlide(e) {
        const bpm = parseInt(e.target.value, 10);
        this._selectedTempo = bpm;
        this._updateTempoDisplay(bpm);

        clearTimeout(this._tempoDebounce);
        this._tempoDebounce = setTimeout(() => {
            const segId = this.currentSegment.segment_id || this.currentSegment.id;
            this._loadReferenceAtTempo(segId, bpm);
        }, 500);
    }

    _resetTempo() {
        const defaultBpm = Math.round(this._originalTempo * 0.7 / 5) * 5;
        document.getElementById('tempo-slider').value = defaultBpm;
        this._selectedTempo = defaultBpm;
        this._updateTempoDisplay(defaultBpm);
        const segId = this.currentSegment.segment_id || this.currentSegment.id;
        this._loadReferenceAtTempo(segId, defaultBpm);
    }

    async _loadReferenceAtTempo(segmentId, bpm) {
        if (this._isLoadingTempo) return;
        this._isLoadingTempo = true;

        const listenBtn = document.getElementById('listen-btn');
        const savedText = listenBtn.textContent;
        listenBtn.textContent = 'Loading...';
        listenBtn.disabled = true;

        try {
            const formData = new FormData();
            formData.append('segment_id', segmentId);
            formData.append('bpm', bpm.toString());

            const res = await fetch('/api/render-tempo', { method: 'POST', body: formData });
            if (!res.ok) {
                const errText = await res.text().catch(() => '');
                throw new Error(`Render failed (${res.status}): ${errText}`);
            }

            const blob = await res.blob();
            const url = URL.createObjectURL(blob);

            const audio = document.getElementById('reference-audio');
            if (audio.src.startsWith('blob:')) URL.revokeObjectURL(audio.src);
            audio.src = url;
        } catch (err) {
            console.error('Tempo render failed:', err);
        } finally {
            listenBtn.textContent = savedText;
            listenBtn.disabled = false;
            this._isLoadingTempo = false;
        }
    }

    // ── Metronome ──

    _toggleMetronome() {
        const btn = document.getElementById('metronome-btn');
        if (this._metronomeOn) {
            this._stopMetronome();
            btn.textContent = 'Metronome';
            btn.classList.remove('active');
        } else {
            this._startMetronome(this._selectedTempo);
            btn.textContent = 'Stop Click';
            btn.classList.add('active');
        }
        this._metronomeOn = !this._metronomeOn;
    }

    _startMetronome(bpm) {
        this._stopMetronome();
        this._metronomeCtx = new (window.AudioContext || window.webkitAudioContext)();
        const intervalMs = (60 / bpm) * 1000;
        let beat = 0;

        const timeSig = this.currentSegment?.time_signature;
        const beatsPerBar = (timeSig && timeSig[0]) || 4;

        const click = () => {
            const osc = this._metronomeCtx.createOscillator();
            const gain = this._metronomeCtx.createGain();
            osc.connect(gain);
            gain.connect(this._metronomeCtx.destination);

            const isDownbeat = beat % beatsPerBar === 0;
            osc.frequency.value = isDownbeat ? 1000 : 800;
            gain.gain.value = isDownbeat ? 0.3 : 0.15;

            osc.start();
            gain.gain.exponentialRampToValueAtTime(
                0.001, this._metronomeCtx.currentTime + 0.05
            );
            osc.stop(this._metronomeCtx.currentTime + 0.05);
            beat++;
        };

        click();
        this._metronomeInterval = setInterval(click, intervalMs);
    }

    _stopMetronome() {
        if (this._metronomeInterval) {
            clearInterval(this._metronomeInterval);
            this._metronomeInterval = null;
        }
        if (this._metronomeCtx) {
            this._metronomeCtx.close();
            this._metronomeCtx = null;
        }
    }

    // ── Reference playback ──

    _toggleReference() {
        const audio = document.getElementById('reference-audio');
        const btn = document.getElementById('listen-btn');

        if (audio.paused) {
            audio.play();
            btn.textContent = 'Playing...';
            audio.onended = () => { btn.textContent = 'Listen to Reference'; };
        } else {
            audio.pause();
            audio.currentTime = 0;
            btn.textContent = 'Listen to Reference';
        }
    }

    // ── Recording flow ──

    async startRecording() {
        if (this.currentMode === 'daily' && this._dailyUserScore != null) return;

        if (this._metronomeOn) {
            this._stopMetronome();
            this._metronomeOn = false;
            const btn = document.getElementById('metronome-btn');
            if (btn) { btn.textContent = 'Metronome'; btn.classList.remove('active'); }
        }

        const ok = await this.recorder.setup();
        if (!ok) {
            alert('Could not access microphone. Please check permissions.');
            return;
        }

        document.getElementById('record-btn').classList.add('hidden');
        document.getElementById('results-section').classList.add('hidden');

        const countdownEl = document.getElementById('countdown-container');
        const numberEl = document.getElementById('countdown-number');
        countdownEl.classList.remove('hidden');

        let count = 3;
        numberEl.textContent = count;

        await new Promise((resolve) => {
            const iv = setInterval(() => {
                count--;
                if (count > 0) {
                    numberEl.textContent = count;
                } else {
                    clearInterval(iv);
                    countdownEl.classList.add('hidden');
                    resolve();
                }
            }, 1000);
        });

        this.recorder.start();
        this.isRecording = true;
        this.recordingStartTime = Date.now();

        document.getElementById('recording-status').classList.remove('hidden');
        this._startTimer();
    }

    async stopRecording() {
        if (!this.isRecording) return;
        this.isRecording = false;
        this._stopTimer();

        document.getElementById('recording-status').classList.add('hidden');
        this._showLoading('Analyzing your performance...');

        const { wavBlob, rawBlob } = await this.recorder.stop();
        this.recorder.cleanup();

        this._playbackBlob = rawBlob;
        await this._submitForScoring(wavBlob);
    }

    tryAgain() {
        document.getElementById('results-section').classList.add('hidden');
        document.getElementById('record-btn').classList.remove('hidden');
    }

    // ── Timer ──

    _startTimer() {
        const timerEl = document.getElementById('recording-timer');
        this.recordingTimer = setInterval(() => {
            const elapsed = Math.floor((Date.now() - this.recordingStartTime) / 1000);
            const m = Math.floor(elapsed / 60);
            const s = elapsed % 60;
            timerEl.textContent = `${m}:${s.toString().padStart(2, '0')}`;
        }, 1000);
    }

    _stopTimer() {
        if (this.recordingTimer) {
            clearInterval(this.recordingTimer);
            this.recordingTimer = null;
        }
    }

    // ── Scoring ──

    async _submitForScoring(audioBlob) {
        try {
            const formData = new FormData();
            const segId = this.currentSegment.segment_id || this.currentSegment.id;
            formData.append('segment_id', segId);
            formData.append('audio', audioBlob, 'recording.wav');

            const res = await fetch('/api/score', { method: 'POST', body: formData });

            if (!res.ok) {
                const errData = await res.json().catch(() => ({}));
                throw new Error(errData.error || 'Scoring failed');
            }

            const result = await res.json();
            this._hideLoading();
            this._displayResults(result);
        } catch (err) {
            this._hideLoading();
            this._showError(`Couldn't analyze your performance: ${err.message}`);
            document.getElementById('record-btn').classList.remove('hidden');
            console.error(err);
        }
    }

    _inflateScore(raw) {
        return Math.pow(raw, 0.7);
    }

    // ── Results display ──

    _displayResults(result) {
        const { scores } = result;

        const displayed = this._inflateScore(scores.overall);
        document.getElementById('overall-score-value').textContent =
            `${Math.round(displayed * 100)}%`;

        setTimeout(() => {
            this._animateBar('pitch', scores.pitch);
            this._animateBar('rhythm', scores.rhythm);
            this._animateBar('completeness', scores.completeness);
        }, 100);

        if (this._playbackBlob) {
            const playbackSection = document.getElementById('recording-playback');
            const playbackAudio = document.getElementById('playback-audio');
            playbackAudio.src = URL.createObjectURL(this._playbackBlob);
            playbackSection.classList.remove('hidden');
        }

        document.getElementById('results-section').classList.remove('hidden');

        if (this.currentMode === 'daily') {
            document.getElementById('try-again-btn').classList.add('hidden');
            this._submitToLeaderboard(result);
        } else {
            document.getElementById('try-again-btn').classList.remove('hidden');
        }
    }

    _animateBar(category, score) {
        const pct = Math.round(score * 100);
        const bar = document.getElementById(`${category}-bar`);
        const val = document.getElementById(`${category}-value`);
        if (bar) bar.style.width = `${pct}%`;
        if (val) val.textContent = `${pct}%`;
    }

    // ── Daily persistence ──

    _saveDailyResult(challengeNumber, scores, userScore) {
        localStorage.setItem('sightreadle_daily', JSON.stringify({
            challenge: challengeNumber,
            scores,
            userScore,
        }));
    }

    _getSavedDailyResult(challengeNumber) {
        try {
            const raw = localStorage.getItem('sightreadle_daily');
            if (!raw) return null;
            const data = JSON.parse(raw);
            if (data.challenge !== challengeNumber) return null;
            return data;
        } catch { return null; }
    }

    _restoreSavedResult(saved) {
        const { scores, userScore } = saved;

        document.getElementById('overall-score-value').textContent =
            `${Math.round(userScore * 100)}%`;

        const bar = (cat, val) => {
            const b = document.getElementById(`${cat}-bar`);
            const v = document.getElementById(`${cat}-value`);
            if (b) b.style.width = `${Math.round(val * 100)}%`;
            if (v) v.textContent = `${Math.round(val * 100)}%`;
        };
        bar('pitch', scores.pitch);
        bar('rhythm', scores.rhythm);
        bar('completeness', scores.completeness);

        document.getElementById('results-section').classList.remove('hidden');
        document.getElementById('record-btn').classList.add('hidden');
        document.getElementById('try-again-btn').classList.add('hidden');
    }

    // ── Leaderboard ──

    _getUserId() {
        let userId = localStorage.getItem('sightreadle_user_id');
        if (!userId) {
            userId = 'anon_' + Math.random().toString(36).substring(2, 10);
            localStorage.setItem('sightreadle_user_id', userId);
        }
        return userId;
    }

    async _submitToLeaderboard(scoringResult) {
        try {
            const res = await fetch('/api/daily-score', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    user_id: this._getUserId(),
                    score: this._inflateScore(scoringResult.scores.overall),
                    n_correct: scoringResult.summary.correct,
                    n_missed: scoringResult.summary.missed,
                    n_extra: scoringResult.summary.extra || 0,
                    total_expected: scoringResult.summary.total_expected,
                }),
            });

            const data = await res.json();
            this._dailyUserScore = data.your_score;
            this._displayLeaderboard(data);

            if (this.currentSegment?.challenge_number != null) {
                this._saveDailyResult(
                    this.currentSegment.challenge_number,
                    scoringResult.scores,
                    this._inflateScore(scoringResult.scores.overall),
                );
            }
        } catch (err) {
            console.error('Leaderboard error:', err);
        }
    }

    async _loadDailyHistogram() {
        try {
            const res = await fetch('/api/daily-leaderboard');
            const data = await res.json();

            const section = document.getElementById('leaderboard-section');
            section.classList.remove('hidden');

            document.getElementById('leaderboard-message').textContent =
                `${data.total_players} players today`;
            document.getElementById('leaderboard-stats').innerHTML =
                `Top: <strong>${Math.round(data.top_score * 100)}%</strong>` +
                ` \u00B7 Median: <strong>${Math.round(data.median_score * 100)}%</strong>`;

            this._drawHistogram(data.distribution, this._dailyUserScore);
        } catch (err) {
            console.error('Histogram load error:', err);
        }
    }

    _displayLeaderboard(data) {
        const section = document.getElementById('leaderboard-section');
        section.classList.remove('hidden');

        const message = document.getElementById('leaderboard-message');
        if (data.rank === 1) {
            message.textContent = `You're #1 out of ${data.total_players} players today!`;
        } else if (data.rank === data.total_players) {
            message.textContent = `Rank #${data.rank} of ${data.total_players} players today. Keep practicing!`;
        } else {
            message.textContent =
                `You scored better than ${data.percentile}% of ${data.total_players} players today!`;
        }

        document.getElementById('leaderboard-stats').innerHTML =
            `Rank: <strong>#${data.rank}</strong> of ${data.total_players}` +
            ` \u00B7 Top: <strong>${Math.round(data.top_score * 100)}%</strong>` +
            ` \u00B7 Median: <strong>${Math.round(data.median_score * 100)}%</strong>`;

        this._drawHistogram(data.distribution, data.your_score);
    }

    _drawHistogram(distribution, userScore) {
        const canvas = document.getElementById('distribution-chart');
        if (!canvas || !distribution) return;

        const ctx = canvas.getContext('2d');
        const w = canvas.width;
        const h = canvas.height;
        const barW = w / distribution.length;
        const maxCount = Math.max(...distribution.map(d => d.count), 1);

        const userBucket = userScore != null
            ? Math.min(Math.floor(userScore * 10), 9) : -1;

        ctx.clearRect(0, 0, w, h);

        distribution.forEach((bucket, i) => {
            const barH = (bucket.count / maxCount) * (h - 25);
            const x = i * barW;
            const y = h - barH - 20;

            ctx.fillStyle = i === userBucket ? '#e53e3e' : '#667eea';
            ctx.fillRect(x + 2, y, barW - 4, barH);

            if (i === userBucket) {
                ctx.fillStyle = '#e53e3e';
                ctx.font = 'bold 8px sans-serif';
                ctx.textAlign = 'center';
                ctx.fillText('YOU', x + barW / 2, y - 3);
            }

            ctx.fillStyle = '#888';
            ctx.font = '9px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText(`${i * 10}%`, x + barW / 2, h - 5);
        });
    }

    // ── Helpers ──

    _formatPieceName(raw) {
        if (!raw) return 'Unknown Piece';
        return raw
            .replace(/\.(mid|musicxml|mxl|xml)$/i, '')
            .replace(/_/g, ' ')
            .replace(/\b\w/g, (c) => c.toUpperCase());
    }

    _showLoading(msg) {
        document.getElementById('loading-message').textContent = msg;
        document.getElementById('loading-overlay').classList.remove('hidden');
    }

    _hideLoading() {
        document.getElementById('loading-overlay').classList.add('hidden');
    }

    _showError(msg) {
        const card = document.getElementById('challenge-card');
        const existing = card.querySelector('.error');
        if (existing) existing.remove();

        const div = document.createElement('div');
        div.className = 'error';
        div.textContent = msg;
        card.prepend(div);

        setTimeout(() => div.remove(), 8000);
    }
}
