# ScribeAI: Your AI-Powered Transcription and Note-Taking Assistant

ScribeAI transforms your spoken words into structured, polished, and ready-to-use notes. Whether you're a doctor dictating patient notes, a student recording a lecture, or just journaling your thoughts, ScribeAI listens, transcribes, and reformats your audio into the perfect format for your needs.

Powered by Google's Gemini API, it offers a seamless experience from recording to final document, right in your browser.

## Key Features

*   **🎙️ Live Recording with Laps:** Record audio directly in the app. For long sessions like meetings or lectures, use the **Lap** button to process audio in chunks without ever stopping the recording.
*   **⬆️ Audio File Upload:** Have an existing audio file? Upload common formats (MP3, WAV, M4A, etc.) and let ScribeAI transcribe and polish it for you.
*   **🧠 Specialized Note Modes:** Choose the right assistant for the job:
    *   **👩‍⚕️ Doctor's Note:** Automatically formats transcriptions into a structured SOAP note, complete with patient details.
    *   **📔 Personal Journal:** Transforms your stream-of-consciousness into an organized journal entry with identified themes and questions to ponder.
    *   **🎓 Study Notes:** Organizes lecture content into core principles, key takeaways, and points of confusion.
    *   **⚙️ Custom Instructions:** Define your own rules! Tell the AI exactly how to format your text, from simple summaries to complex reports.
*   **📄 Dual-View Editor:** Instantly switch between the **Polished** note (beautifully formatted in markdown) and the **Raw** transcription (the original, word-for-word text).
*   **📊 Detailed Metadata & Cost Tracking:** Every note includes its timestamp, duration, audio file size, processing mode, and an **estimated cost** for the AI processing, giving you full transparency.
*   **✨ Rich Export & Copy Options:**
    *   Download your polished note as a standard **Markdown (.md) file**.
    *   Download the complete **recorded audio**.
    *   Copy the polished note, raw text, or metadata to your clipboard with a single click.
*   **🎨 User-Friendly Interface:**
    *   **Light & Dark Themes:** Switch themes to match your preference.
    *   **Timezone Support:** Timestamps are localized to your selected timezone.
    *   **Responsive Design:** Works beautifully on both desktop and mobile devices.
    *   **Auto-Download:** Automatically save your polished note as soon as it's ready.

## How to Use ScribeAI

1.  **Select Your Mode:** From the dropdown menu, choose the type of note you want to create (e.g., "Doctor's Note").
2.  **Record or Upload:**
    *   **To Record:** Click the big microphone button. Use the pause/resume, lap, and stop controls as needed.
    *   **To Upload:** Click the upload icon in the header and select an audio file from your device.
3.  **Review Your Notes:** The app will first show the raw, verbatim transcription. Shortly after, the "Polished" tab will update with the structured note formatted according to your chosen mode.
4.  **Export Your Work:** Use the icons in the top-right corner to download your note/audio or copy the content to your clipboard.

---

## Run Locally

**Prerequisites:**  Node.js

1. Install dependencies:
   `npm install`
2. Set the `GEMINI_API_KEY` in [.env.local](.env.local) to your Gemini API key
3. Run the app:
   `npm run dev`
