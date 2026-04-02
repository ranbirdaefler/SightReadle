/**
 * AudioRecorder — captures microphone audio and produces both a WAV blob
 * (for server-side scoring) and keeps the raw browser blob (for playback).
 */
class AudioRecorder {
    constructor() {
        this.audioContext = null;
        this.mediaRecorder = null;
        this.recordedChunks = [];
        this.stream = null;
        this.wavBlob = null;
        this.rawBlob = null;
    }

    async setup() {
        try {
            this.stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    sampleRate: 44100,
                    channelCount: 1,
                    echoCancellation: false,
                    noiseSuppression: false,
                    autoGainControl: false,
                },
            });

            this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: 44100,
            });

            return true;
        } catch (err) {
            console.error('Microphone setup failed:', err);
            return false;
        }
    }

    start() {
        this.recordedChunks = [];
        this.wavBlob = null;
        this.rawBlob = null;

        this.mediaRecorder = new MediaRecorder(this.stream, {
            mimeType: this._pickMimeType(),
        });

        this.mediaRecorder.ondataavailable = (e) => {
            if (e.data.size > 0) this.recordedChunks.push(e.data);
        };

        this.mediaRecorder.start(100);
    }

    stop() {
        return new Promise((resolve) => {
            this.mediaRecorder.onstop = async () => {
                this.rawBlob = new Blob(this.recordedChunks, {
                    type: this.mediaRecorder.mimeType,
                });

                try {
                    this.wavBlob = await this._convertToWav(this.rawBlob);
                } catch (err) {
                    console.error('WAV conversion failed, using raw blob for server too:', err);
                    this.wavBlob = this.rawBlob;
                }

                resolve({ wavBlob: this.wavBlob, rawBlob: this.rawBlob });
            };
            this.mediaRecorder.stop();
        });
    }

    cleanup() {
        if (this.stream) {
            this.stream.getTracks().forEach((t) => t.stop());
            this.stream = null;
        }
    }

    _pickMimeType() {
        const preferred = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg', 'audio/mp4'];
        for (const mt of preferred) {
            if (MediaRecorder.isTypeSupported(mt)) return mt;
        }
        return '';
    }

    async _convertToWav(blob) {
        if (!this.audioContext || this.audioContext.state === 'closed') {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: 44100,
            });
        }

        const arrayBuffer = await blob.arrayBuffer();
        const audioBuffer = await this.audioContext.decodeAudioData(arrayBuffer);
        return this._audioBufferToWav(audioBuffer);
    }

    _audioBufferToWav(buffer) {
        const numChannels = 1;
        const sampleRate = buffer.sampleRate;
        const channelData = buffer.getChannelData(0);
        const length = channelData.length;
        const dataSize = length * 2;
        const headerSize = 44;
        const arrayBuffer = new ArrayBuffer(headerSize + dataSize);
        const view = new DataView(arrayBuffer);

        const writeStr = (offset, str) => {
            for (let i = 0; i < str.length; i++) {
                view.setUint8(offset + i, str.charCodeAt(i));
            }
        };

        writeStr(0, 'RIFF');
        view.setUint32(4, 36 + dataSize, true);
        writeStr(8, 'WAVE');
        writeStr(12, 'fmt ');
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true);
        view.setUint16(22, numChannels, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * numChannels * 2, true);
        view.setUint16(32, numChannels * 2, true);
        view.setUint16(34, 16, true);
        writeStr(36, 'data');
        view.setUint32(40, dataSize, true);

        let offset = 44;
        for (let i = 0; i < length; i++) {
            const sample = Math.max(-1, Math.min(1, channelData[i]));
            view.setInt16(offset, sample * 0x7fff, true);
            offset += 2;
        }

        return new Blob([arrayBuffer], { type: 'audio/wav' });
    }
}
