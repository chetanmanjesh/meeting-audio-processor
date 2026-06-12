# meeting-audio-processor

Standalone CLI to turn a meeting audio file into a diarized transcript + structured minutes of meeting (MoM). Built as a sandbox before integrating into Studio Portal.

## Pipeline

1. **Deepgram** (`nova-3`) — transcription + speaker diarization in one API call.
2. **Claude** (`claude-sonnet-4-6`) — extracts summary, decisions, action items, open questions, per-speaker talking points as JSON.

Speaker identity is intentionally out of scope — speakers stay as `Speaker 0`, `Speaker 1`, etc. (Claude will pick up names if someone is addressed by name in the audio.)

## Setup

```bash
cd ~/Desktop/dashboard_peggy/meeting-audio-processor
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your two API keys
```

**API keys:**
- Deepgram: https://console.deepgram.com/signup ($200 free credit, ~$0.0043/min for nova-3)
- Anthropic: https://console.anthropic.com/

## Usage

```bash
python process.py path/to/meeting.mp3
```

Output (defaults to `./output/`):
- `<name>.transcript.txt` — diarized transcript
- `<name>.mom.json` — structured MoM data
- `<name>.mom.md` — human-readable MoM
- `<name>.deepgram.json` — raw Deepgram response (for debugging)

Custom output dir:
```bash
python process.py meeting.mp3 --out ~/Desktop/meeting1/
```

## Notes

- Sync only — the script blocks until both API calls finish. Fine for testing; switch to a job queue when integrating into the portal.
- Supports any audio format Deepgram accepts (mp3, wav, m4a, mp4, flac, ogg, etc.).
- Cost per hour of audio: ~$0.26 Deepgram + ~$0.03 Claude = ~$0.30.
