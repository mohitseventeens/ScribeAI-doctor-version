/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/
/* tslint:disable */

import {GoogleGenAI} from '@google/genai';
import {marked} from 'marked';

// ============================================================================
// NEW: Waveform Visualization Module
// ============================================================================

interface WaveformConfig {
  fftSize?: number;
  smoothingTimeConstant?: number;
  lineWidth?: number;
  waveColor?: string;
  backgroundColor?: string;
}

/**
 * Handles all Web Audio API logic for analyzing a MediaStream.
 */
class AudioAnalyzer {
  private audioContext: AudioContext | null = null;
  private analyserNode: AnalyserNode | null = null;
  private dataArray: Uint8Array | null = null;
  private source: MediaStreamAudioSourceNode | null = null;

  constructor(private config: WaveformConfig = {}) {}

  public async initialize(stream: MediaStream): Promise<boolean> {
    try {
      if (this.audioContext) {
        return true; // Already initialized
      }
      this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
      this.source = this.audioContext.createMediaStreamSource(stream);
      this.analyserNode = this.audioContext.createAnalyser();

      this.analyserNode.fftSize = this.config.fftSize || 2048;
      this.analyserNode.smoothingTimeConstant = this.config.smoothingTimeConstant || 0.6;

      this.dataArray = new Uint8Array(this.analyserNode.fftSize);
      
      this.source.connect(this.analyserNode);
      return true;
    } catch (error) {
      console.error('Failed to initialize audio analyzer:', error);
      this.cleanup();
      return false;
    }
  }

  public getWaveformData(): Uint8Array | null {
    if (!this.analyserNode || !this.dataArray) return null;
    this.analyserNode.getByteTimeDomainData(this.dataArray);
    return this.dataArray;
  }

  public cleanup(): void {
    this.source?.disconnect();
    if (this.audioContext?.state !== 'closed') {
      this.audioContext?.close().catch(err => console.error("Error closing AudioContext:", err));
    }
    this.audioContext = null;
    this.analyserNode = null;
    this.source = null;
    this.dataArray = null;
  }
}

/**
 * Handles all Canvas drawing logic for the waveform.
 */
class WaveformRenderer {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private animationId: number | null = null;

  constructor(
    canvas: HTMLCanvasElement, 
    private config: WaveformConfig,
    private state: { isActive: boolean; isPaused: boolean }
  ) {
    this.canvas = canvas;
    const context = canvas.getContext('2d');
    if (!context) throw new Error('Could not get 2D context from canvas');
    this.ctx = context;
    this.resize();
  }
  
  public resize(): void {
    const dpr = window.devicePixelRatio || 1;
    const rect = this.canvas.getBoundingClientRect();
    this.canvas.width = Math.round(rect.width * dpr);
    this.canvas.height = Math.round(rect.height * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  private clearCanvas(): void {
    this.ctx.clearRect(0, 0, this.canvas.clientWidth, this.canvas.clientHeight);
  }

  private draw(dataArray: Uint8Array): void {
    const { clientWidth: width, clientHeight: height } = this.canvas;
    const dataLength = dataArray.length;
    const sliceWidth = width / dataLength;

    this.ctx.lineWidth = this.config.lineWidth || 2;
    this.ctx.strokeStyle = this.config.waveColor || 'lime';
    this.ctx.beginPath();

    let x = 0;
    for (let i = 0; i < dataLength; i++) {
      const v = dataArray[i] / 128.0;
      const y = (v * height) / 2;
      if (i === 0) {
        this.ctx.moveTo(x, y);
      } else {
        this.ctx.lineTo(x, y);
      }
      x += sliceWidth;
    }

    this.ctx.lineTo(width, height / 2);
    this.ctx.stroke();
  }

  public startAnimationLoop(dataProvider: () => Uint8Array | null): void {
    const animate = () => {
      if (!this.state.isActive) {
        this.stopAnimationLoop();
        return;
      }
      
      this.animationId = requestAnimationFrame(animate);

      if (this.state.isPaused) return;
        
      const data = dataProvider();
      if (data) {
        this.clearCanvas();
        this.draw(data);
      }
    };
    animate();
  }

  public stopAnimationLoop(): void {
    if (this.animationId) {
      cancelAnimationFrame(this.animationId);
      this.animationId = null;
    }
    this.clearCanvas();
  }
}

/**
 * Orchestrates AudioAnalyzer and WaveformRenderer to create a complete visualization.
 */
class WaveformVisualizer {
  private audioAnalyzer: AudioAnalyzer;
  private renderer: WaveformRenderer;
  private state = { isActive: false, isPaused: false };
  
  constructor(
    private canvas: HTMLCanvasElement,
    private config: WaveformConfig = {}
  ) {
    this.audioAnalyzer = new AudioAnalyzer(this.config);
    this.renderer = new WaveformRenderer(this.canvas, this.config, this.state);
  }

  public async start(stream: MediaStream): Promise<boolean> {
    const success = await this.audioAnalyzer.initialize(stream);
    if (!success) return false;

    this.state.isActive = true;
    this.state.isPaused = false;
    this.renderer.startAnimationLoop(() => this.audioAnalyzer.getWaveformData());
    return true;
  }

  public pause(): void { this.state.isPaused = true; }
  public resume(): void { this.state.isPaused = false; }
  
  public stop(): void {
    this.state.isActive = false; // This will cause the animation loop to stop itself.
    this.audioAnalyzer.cleanup();
  }
  
  public resize(): void {
    this.renderer.resize();
  }

  public updateConfig(newConfig: Partial<WaveformConfig>): void {
    Object.assign(this.config, newConfig);
  }
}

// ============================================================================

const MODEL_NAME = 'gemini-2.5-flash';
const COST_PER_1K_PROMPT_TOKENS = 0.000125; // gemini-2.5-flash input
const COST_PER_1K_COMPLETION_TOKENS = 0.000250; // gemini-2.5-flash output

// Mode definitions
type ModeID = 'doctor' | 'journal' | 'learning' | 'custom';

interface Mode {
  id: ModeID;
  name: string;
  instructions: string;
}

const DOCTOR_MODE_INSTRUCTIONS = `You are a medical scribe assisting a doctor. Your task is to transform a raw, transcribed conversation with a patient into a structured clinical note. Your output must be in markdown.

First, populate the patient details table below. Extract the information from the conversation. If a detail is not mentioned, leave the placeholder text (e.g., \`[MR. NO]\`) as is. The **Department** is "Physiotherapy" by default unless specified otherwise.

| | | | |
| :--- | :--- | :--- | :--- |
| **MR. NO:** | [MR. NO] | **Date:** | [Date] |
| **Patient Name:** | [Patient Name] | **O.P.D. NO:** | [O.P.D. NO] |
| **Age:** | [Age] | **Sex:** | [Sex] |
| **Department:** | Physiotherapy | **Occupation:** | [Occupation] |
| **Address:** | [Address] | **Referred Name:** | [Referred Name] |
| **Attendant Name:** | [Attendant Name] | **Final Diagnosis:** | [Final Diagnosis] |

---

After the table, structure the rest of the note using the SOAP format:

## S (Subjective)
- Document the patient's chief complaint and history of present illness in their own words.
- Include relevant details like onset, location, duration, character, aggravating/relieving factors, timing, and severity (the "OLDCARTS" mnemonic).
- List any patient-reported symptoms.

## O (Objective)
- Summarize any objective findings mentioned by the doctor (e.g., "On examination...", "Vitals are...").
- If no objective data is provided, state "No objective findings mentioned."

## A (Assessment)
- List the doctor's diagnoses or differential diagnoses.
- Capture the doctor's reasoning or summary of the patient's condition.

## P (Plan)
- Detail the treatment plan, including:
  - Medications prescribed (with dosage and frequency if mentioned).
  - Diagnostic tests ordered (e.g., labs, imaging).
  - Referrals to specialists.
  - Patient education and counseling points.
  - Follow-up instructions.`;

const MODES: Record<ModeID, Mode> = {
  doctor: {
    id: 'doctor',
    name: "Doctor's Note",
    instructions: DOCTOR_MODE_INSTRUCTIONS,
  },
  journal: {
    id: 'journal',
    name: 'Personal Journal',
    instructions: `You are a reflective journaling partner. Your task is to transform a raw, first-person transcription of an inner monologue into a clear and organized personal journal entry. Your output must be in markdown.
    
# PROCESS:

## 1.  Identify Core Themes: 
Analyze the monologue to find 2-4 main topics or recurring themes. These will be your main headings in markdown (e.g., \`## Reflections on Today's Progress\`).

## 2.  Organize and Summarize: 
Group related thoughts under the appropriate theme. Summarize key points in concise, bulleted lists or short paragraphs, rewriting for clarity while preserving the original meaning.

## 3.  Preserve the Voice: 
Maintain the first-person ("I," "my," "me") perspective. The output should feel like a personal reflection.

## 4.  Extract Key Questions: 
If the monologue contains self-directed questions (e.g., "What should I do about...?"), collect them into a final section titled \`## Questions to Ponder\`.`,
  },
  learning: {
    id: 'learning',
    name: 'Study Notes',
    instructions: `You are a student organizing study notes from a lecture or study session. Structure the output in markdown.

# Study Notes: [Insert Topic]

## 1. Core Principles
Summarize the fundamental concepts and main ideas that were covered.

## 2. Key Takeaways
Use a bulleted list for specific facts, formulas, or important "Aha!" moments.

## 3. Points of Confusion
List any questions or topics that remained unclear and require further review.

## 4. Connections
Note how this topic connects to other subjects or your personal knowledge.`,
  },
  custom: {
    id: 'custom',
    name: 'Custom Instructions',
    instructions: '', // This will be dynamically updated from localStorage.
  },
};

const TIMEZONES = [
  'UTC', 'Warsaw', 'Aurangabad', 'Pune', 'Mumbai'
];

interface Note {
  id: string;
  rawTranscription: string;
  polishedNote: string;
  timestamp: number;
  duration: number; // in ms
  audioSize: number; // in bytes
  modeId: ModeID;
  promptTokens: number;
  completionTokens: number;
  cost: number;
}

class VoiceNotesApp {
  private genAI: any = null;
  private mediaRecorder: MediaRecorder | null = null;
  private newButton: HTMLButtonElement;
  private uploadButton: HTMLButtonElement;
  private downloadAudioButton: HTMLButtonElement;
  private downloadNoteButton: HTMLButtonElement;
  private audioUploadInput: HTMLInputElement;
  private themeToggleButton: HTMLButtonElement;
  private copyButton: HTMLButtonElement;
  private copyMetaButton: HTMLButtonElement;
  private copyRawButton: HTMLButtonElement;
  private themeToggleIcon: HTMLElement;
  private editCustomPromptButton: HTMLButtonElement;
  private audioChunks: Blob[] = [];
  private sessionAudioChunks: Blob[] = [];
  private sessionMimeType: string = '';
  
  // Recording State
  private isRecording = false;
  private isPaused = false;
  private isProcessing = false;
  private stopReason: 'stop' | 'lap' | null = null;
  private lapCount = 0;
  private allRawLapText = '';
  private totalDurationMs = 0;

  private currentNote: Note | null = null;
  private stream: MediaStream | null = null;
  private currentModeId: ModeID = 'doctor';

  // UI Elements for Mode Selector
  private modeSelectorContainer: HTMLDivElement;
  private modeSelectorButton: HTMLButtonElement;
  private currentModeNameSpan: HTMLSpanElement;
  private modeList: HTMLDivElement;

  // UI Elements for Timezone Selector
  private timezoneSelectorContainer: HTMLDivElement;
  private timezoneSelectorButton: HTMLButtonElement;
  private currentTimezoneNameSpan: HTMLSpanElement;
  private timezoneList: HTMLDivElement;
  private currentTimezone: string = 'Aurangabad';
  
  // Custom Prompt Modal
  private customPromptModal: HTMLDivElement;
  private customPromptTextarea: HTMLTextAreaElement;
  private saveCustomPromptButton: HTMLButtonElement;
  private cancelCustomPromptButton: HTMLButtonElement;
  private customPromptInstructions: string = '';
  private customPromptSettingsItem: HTMLDivElement;

  // Info Modal
  private infoModal: HTMLDivElement;
  private infoModalTitle: HTMLHeadingElement;
  private infoModalContent: HTMLDivElement;
  private infoModalCloseButton: HTMLButtonElement;

  // Settings Menu
  private settingsMenuContainer: HTMLDivElement;
  private settingsButton: HTMLButtonElement;
  private settingsMenuList: HTMLDivElement;
  
  // Metadata display
  private noteMetadata: HTMLDivElement;
  private metaDatetime: HTMLDivElement;
  private metaDuration: HTMLDivElement;
  private metaSize: HTMLDivElement;
  private metaMode: HTMLDivElement;
  private metaCost: HTMLDivElement;

  // Live recording UI
  private fabRecord: HTMLButtonElement;
  private recordingDialog: HTMLDivElement;
  private liveRecordingTitle: HTMLDivElement;
  private liveWaveformCanvas: HTMLCanvasElement;
  private liveRecordingTimerDisplay: HTMLDivElement;

  // Recording Controls
  private stopButton: HTMLButtonElement;
  private pauseButton: HTMLButtonElement;
  private lapButton: HTMLButtonElement;

  // Content display
  private rawTranscription: HTMLDivElement;
  private polishedNote: HTMLDivElement;
  private globalStatus: HTMLDivElement;

  // Tab UI
  private tabButtons: NodeListOf<HTMLButtonElement>;
  private tabIndicator: HTMLDivElement;
  private tabPanes: NodeListOf<HTMLDivElement>;
  
  // NEW: Waveform Visualizer
  private waveformVisualizer: WaveformVisualizer | null = null;
  private timerIntervalId: number | null = null;
  private recordingStartTime: number = 0;

  // Mobile UI elements
  private moreMenuContainer: HTMLDivElement;
  private moreMenuButton: HTMLButtonElement;
  private moreMenuList: HTMLDivElement;
  private moreMenuThemeToggleIcon: HTMLElement | null = null;
  private bottomNav: HTMLElement;
  private bottomNavNew: HTMLButtonElement;
  private bottomNavRecord: HTMLButtonElement;
  private bottomNavUpload: HTMLButtonElement;
  private timezoneModal: HTMLDivElement;
  private mobileTimezoneList: HTMLDivElement;
  private timezoneModalCloseButton: HTMLButtonElement;

  // Auto-download feature
  private autoDownloadToggle: HTMLInputElement;
  private autoDownloadEnabled: boolean = false;

  // Security properties
  private pinModal: HTMLDivElement;
  private pinTitle: HTMLHeadingElement;
  private pinSubtitle: HTMLParagraphElement;
  private pinForm: HTMLDivElement;
  private pinInputs: HTMLInputElement[];
  private pinErrorMessage: HTMLDivElement;
  private pinSubmitButton: HTMLButtonElement;
  private pinForgotButton: HTMLButtonElement;
  private pinMode: 'set' | 'enter' | 'confirm' = 'enter';
  private firstPinAttempt = '';
  
  // API Key properties
  private apiKeyModal: HTMLDivElement;
  private apiKeyInput: HTMLInputElement;
  private saveApiKeyButton: HTMLButtonElement;
  private toggleApiKeyVisibilityButton: HTMLButtonElement;
  private updateApiKeyButton: HTMLButtonElement;

  constructor() {
    // Main buttons
    this.newButton = document.getElementById('newButton') as HTMLButtonElement;
    this.uploadButton = document.getElementById('uploadButton') as HTMLButtonElement;
    this.downloadAudioButton = document.getElementById('downloadAudioButton') as HTMLButtonElement;
    this.downloadNoteButton = document.getElementById('downloadNoteButton') as HTMLButtonElement;
    this.audioUploadInput = document.getElementById('audioUploadInput') as HTMLInputElement;
    this.themeToggleButton = document.getElementById('themeToggleButton') as HTMLButtonElement;
    this.copyButton = document.getElementById('copyButton') as HTMLButtonElement;
    this.copyMetaButton = document.getElementById('copyMetaButton') as HTMLButtonElement;
    this.copyRawButton = document.getElementById('copyRawButton') as HTMLButtonElement;
    this.themeToggleIcon = this.themeToggleButton.querySelector('i') as HTMLElement;
    this.editCustomPromptButton = document.getElementById('editCustomPromptButton') as HTMLButtonElement;
    this.autoDownloadToggle = document.getElementById('autoDownloadToggle') as HTMLInputElement;

    // Settings Menu
    this.settingsMenuContainer = document.getElementById('settingsMenuContainer') as HTMLDivElement;
    this.settingsButton = document.getElementById('settingsButton') as HTMLButtonElement;
    this.settingsMenuList = document.getElementById('settingsMenuList') as HTMLDivElement;
    this.customPromptSettingsItem = document.getElementById('customPromptSettingsItem') as HTMLDivElement;

    // Mobile UI elements
    this.moreMenuContainer = document.getElementById('moreMenuContainer') as HTMLDivElement;
    this.moreMenuButton = document.getElementById('moreMenuButton') as HTMLButtonElement;
    this.moreMenuList = document.getElementById('moreMenuList') as HTMLDivElement;
    this.bottomNav = document.getElementById('bottomNav') as HTMLElement;
    this.bottomNavNew = document.getElementById('bottomNavNew') as HTMLButtonElement;
    this.bottomNavRecord = document.getElementById('bottomNavRecord') as HTMLButtonElement;
    this.bottomNavUpload = document.getElementById('bottomNavUpload') as HTMLButtonElement;
    this.timezoneModal = document.getElementById('timezoneModal') as HTMLDivElement;
    this.mobileTimezoneList = document.getElementById('mobileTimezoneList') as HTMLDivElement;
    this.timezoneModalCloseButton = document.getElementById('timezoneModalCloseButton') as HTMLButtonElement;


    // Recording controls
    this.fabRecord = document.getElementById('fabRecord') as HTMLButtonElement;
    this.recordingDialog = document.getElementById('recordingDialog') as HTMLDivElement;
    this.stopButton = document.getElementById('stopButton') as HTMLButtonElement;
    this.pauseButton = document.getElementById('pauseButton') as HTMLButtonElement;
    this.lapButton = document.getElementById('lapButton') as HTMLButtonElement;
    
    // Content areas
    this.globalStatus = document.getElementById('globalStatus') as HTMLDivElement;
    this.rawTranscription = document.getElementById('rawTranscription') as HTMLDivElement;
    this.polishedNote = document.getElementById('polishedNote') as HTMLDivElement;
    
    // Mode Selector Elements
    this.modeSelectorContainer = document.getElementById('modeSelectorContainer') as HTMLDivElement;
    this.modeSelectorButton = document.getElementById('modeSelectorButton') as HTMLButtonElement;
    this.currentModeNameSpan = document.getElementById('currentModeName') as HTMLSpanElement;
    this.modeList = document.getElementById('modeList') as HTMLDivElement;

    // Timezone Selector Elements
    this.timezoneSelectorContainer = document.getElementById('timezoneSelectorContainer') as HTMLDivElement;
    this.timezoneSelectorButton = document.getElementById('timezoneSelectorButton') as HTMLButtonElement;
    this.currentTimezoneNameSpan = document.getElementById('currentTimezoneName') as HTMLSpanElement;
    this.timezoneList = document.getElementById('timezoneList') as HTMLDivElement;

    // Custom Prompt Modal
    this.customPromptModal = document.getElementById('customPromptModal') as HTMLDivElement;
    this.customPromptTextarea = document.getElementById('customPromptTextarea') as HTMLTextAreaElement;
    this.saveCustomPromptButton = document.getElementById('saveCustomPromptButton') as HTMLButtonElement;
    this.cancelCustomPromptButton = document.getElementById('cancelCustomPromptButton') as HTMLButtonElement;

    // Info Modal
    this.infoModal = document.getElementById('infoModal') as HTMLDivElement;
    this.infoModalTitle = document.getElementById('infoModalTitle') as HTMLHeadingElement;
    this.infoModalContent = document.getElementById('infoModalContent') as HTMLDivElement;
    this.infoModalCloseButton = document.getElementById('infoModalCloseButton') as HTMLButtonElement;
    
    // Metadata
    this.noteMetadata = document.getElementById('noteMetadata') as HTMLDivElement;
    this.metaDatetime = document.getElementById('meta-datetime') as HTMLDivElement;
    this.metaDuration = document.getElementById('meta-duration') as HTMLDivElement;
    this.metaSize = document.getElementById('meta-size') as HTMLDivElement;
    this.metaMode = document.getElementById('meta-mode') as HTMLDivElement;
    this.metaCost = document.getElementById('meta-cost') as HTMLDivElement;

    // Live display
    this.liveRecordingTitle = document.getElementById('liveRecordingTitle') as HTMLDivElement;
    this.liveWaveformCanvas = document.getElementById('liveWaveformCanvas') as HTMLCanvasElement;
    this.liveRecordingTimerDisplay = document.getElementById('liveRecordingTimerDisplay') as HTMLDivElement;

    // Tabs
    this.tabButtons = document.querySelectorAll('.tab-button');
    this.tabIndicator = document.querySelector('.tab-indicator') as HTMLDivElement;
    this.tabPanes = document.querySelectorAll('.tab-pane');
    
    // Get PIN modal elements
    this.pinModal = document.getElementById('pinModal') as HTMLDivElement;
    this.pinTitle = document.getElementById('pinTitle') as HTMLHeadingElement;
    this.pinSubtitle = document.getElementById('pinSubtitle') as HTMLParagraphElement;
    this.pinForm = document.getElementById('pinForm') as HTMLDivElement;
    this.pinInputs = Array.from(this.pinForm.querySelectorAll('.pin-input'));
    this.pinErrorMessage = document.getElementById('pinErrorMessage') as HTMLDivElement;
    this.pinSubmitButton = document.getElementById('pinSubmitButton') as HTMLButtonElement;
    this.pinForgotButton = document.getElementById('pinForgotButton') as HTMLButtonElement;

    // Get API Key modal elements
    this.apiKeyModal = document.getElementById('apiKeyModal') as HTMLDivElement;
    this.apiKeyInput = document.getElementById('apiKeyInput') as HTMLInputElement;
    this.saveApiKeyButton = document.getElementById('saveApiKeyButton') as HTMLButtonElement;
    this.toggleApiKeyVisibilityButton = document.getElementById('toggleApiKeyVisibility') as HTMLButtonElement;
    this.updateApiKeyButton = document.getElementById('updateApiKeyButton') as HTMLButtonElement;

    // Start the security check first.
    this.initSecurity();
  }

  private async initSecurity(): Promise<void> {
    const pinHash = localStorage.getItem('scribeai_pin_hash');
    if (pinHash) {
        let refreshCount = parseInt(localStorage.getItem('scribeai_refresh_count') || '0', 10);
        refreshCount++;
        
        if (refreshCount >= 10) {
            this.showEnterPinScreen(); // This will lead to handlePinSubmit, which will reset the counter on success.
        } else {
            localStorage.setItem('scribeai_refresh_count', String(refreshCount));
            this.unlockApp(); // Just unlock, no PIN needed.
        }
    } else {
        this.showSetPinScreen();
    }
  }
  
  private initializeApp(): void {
    this.bindEventListeners();
    this.initTheme();
    this.initTabs();
    this.initAutoDownload();
    this.initCustomModeSelector();
    this.initTimezoneSelector();
    this.loadCustomPrompt();
    this.loadAndSetInitialMode();
    this.createNewNote();
    this.downloadAudioButton.disabled = true;

    this.initApiKey();
  }

  private initApiKey(): void {
    const apiKey = localStorage.getItem('gemini_api_key');
    if (apiKey) {
      this.initializeGenAI(apiKey);
    } else {
      this.apiKeyModal.style.display = 'flex';
      this.disableAppFeatures();
      this.setGlobalStatus('API Key required to begin', false, true);
    }
  }
  
  private initializeGenAI(apiKey: string): void {
    try {
      this.genAI = new GoogleGenAI({ apiKey });
      this.apiKeyModal.style.display = 'none';
      this.enableAppFeatures();
      this.setGlobalStatus('Ready to record');
    } catch (e) {
      console.error('Error initializing GoogleGenAI:', e);
      this.setGlobalStatus('Failed to initialize AI.', false, true);
      this.disableAppFeatures();
    }
  }

  private disableAppFeatures(): void {
    this.fabRecord.disabled = true;
    this.uploadButton.disabled = true;
    this.bottomNavRecord.disabled = true;
    this.bottomNavUpload.disabled = true;
  }

  private enableAppFeatures(): void {
    this.fabRecord.disabled = false;
    this.uploadButton.disabled = false;
    this.bottomNavRecord.disabled = false;
    this.bottomNavUpload.disabled = false;
  }
  
  private handleSaveApiKey(): void {
    const apiKey = this.apiKeyInput.value.trim();
    if (apiKey) {
      localStorage.setItem('gemini_api_key', apiKey);
      this.initializeGenAI(apiKey);
    } else {
      this.apiKeyInput.reportValidity();
    }
  }

  private handleUpdateApiKey(): void {
    this.apiKeyInput.value = localStorage.getItem('gemini_api_key') || '';
    this.apiKeyModal.style.display = 'flex';
    this.closeSettingsMenu();
    this.closeMoreMenu();
  }

  private toggleApiKeyVisibility(): void {
    const icon = this.toggleApiKeyVisibilityButton.querySelector('i');
    if (!icon) return;
    if (this.apiKeyInput.type === 'password') {
        this.apiKeyInput.type = 'text';
        icon.classList.remove('fa-eye');
        icon.classList.add('fa-eye-slash');
    } else {
        this.apiKeyInput.type = 'password';
        icon.classList.remove('fa-eye-slash');
        icon.classList.add('fa-eye');
    }
  }


  private showSetPinScreen(): void {
    this.pinMode = 'set';
    this.pinModal.style.display = 'flex';
    this.pinTitle.textContent = 'Create a PIN';
    this.pinSubtitle.textContent = 'Set a 4-digit PIN to secure your notes.';
    this.pinSubmitButton.textContent = 'Save PIN';
    this.pinSubmitButton.disabled = true;
    this.pinForgotButton.style.display = 'none';
    this.bindPinEvents();
    this.pinInputs[0].focus();
  }

  private showEnterPinScreen(): void {
    this.pinMode = 'enter';
    this.pinModal.style.display = 'flex';
    this.pinTitle.textContent = 'Enter PIN';
    this.pinSubtitle.textContent = 'Enter your 4-digit PIN to unlock.';
    this.pinSubmitButton.textContent = 'Unlock';
    this.pinSubmitButton.disabled = true;
    this.pinForgotButton.style.display = 'block';
    this.bindPinEvents();
    this.pinInputs[0].focus();
  }

  private async hashPin(pin: string): Promise<string> {
    const salt = 'scribeai-static-salt'; 
    const msgUint8 = new TextEncoder().encode(pin + salt);
    const hashBuffer = await crypto.subtle.digest('SHA-256', msgUint8);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
  }

  private bindPinEvents(): void {
    this.pinInputs.forEach((input, index) => {
        input.addEventListener('input', () => this.handlePinInput(index));
        input.addEventListener('keydown', (e) => this.handlePinBackspace(e, index));
        input.addEventListener('focus', () => input.select());
    });
    this.pinSubmitButton.addEventListener('click', () => this.handlePinSubmit());
    this.pinForgotButton.addEventListener('click', () => this.handleForgotPin());
  }

  private handlePinInput(index: number): void {
    this.pinErrorMessage.style.display = 'none';
    const input = this.pinInputs[index];
    if (input.value && index < this.pinInputs.length - 1) {
        this.pinInputs[index + 1].focus();
    }
    const fullPin = this.pinInputs.map(i => i.value).join('');
    this.pinSubmitButton.disabled = fullPin.length !== 4;
  }
  
  private handlePinBackspace(e: KeyboardEvent, index: number): void {
      if (e.key === 'Backspace' && !this.pinInputs[index].value && index > 0) {
          this.pinInputs[index - 1].focus();
      }
  }

  private async handlePinSubmit(): Promise<void> {
    const pin = this.pinInputs.map(i => i.value).join('');
    if (pin.length !== 4) return;
    
    this.pinSubmitButton.disabled = true;
    
    switch (this.pinMode) {
      case 'set':
        this.firstPinAttempt = pin;
        this.pinMode = 'confirm';
        this.pinTitle.textContent = 'Confirm PIN';
        this.pinSubtitle.textContent = 'Please enter your PIN again.';
        this.pinSubmitButton.textContent = 'Confirm';
        this.clearPinInputs();
        break;
        
      case 'confirm':
        if (pin === this.firstPinAttempt) {
            const pinHash = await this.hashPin(pin);
            localStorage.setItem('scribeai_pin_hash', pinHash);
            localStorage.setItem('scribeai_refresh_count', '0'); // Reset counter
            this.unlockApp();
        } else {
            this.pinErrorMessage.textContent = 'PINs do not match. Please try again.';
            this.pinErrorMessage.style.display = 'block';
            this.pinForm.classList.add('shake');
            setTimeout(() => this.pinForm.classList.remove('shake'), 500);
            
            this.pinMode = 'set';
            this.pinTitle.textContent = 'Create a PIN';
            this.pinSubtitle.textContent = 'Set a 4-digit PIN to secure your notes.';
            this.pinSubmitButton.textContent = 'Save PIN';
            this.clearPinInputs();
        }
        break;
        
      case 'enter':
        const storedHash = localStorage.getItem('scribeai_pin_hash');
        const enteredHash = await this.hashPin(pin);
        if (enteredHash === storedHash) {
            localStorage.setItem('scribeai_refresh_count', '0'); // Reset counter
            this.unlockApp();
        } else {
            this.pinErrorMessage.textContent = 'Incorrect PIN. Please try again.';
            this.pinErrorMessage.style.display = 'block';
            this.pinForm.classList.add('shake');
            setTimeout(() => this.pinForm.classList.remove('shake'), 500);
            this.clearPinInputs();
        }
        break;
    }
  }
  
  private handleForgotPin(): void {
    const confirmation = confirm(
      'Are you sure you want to reset your PIN?\n\nWARNING: This action will erase all your notes and settings. This cannot be undone.'
    );
    if (confirmation) {
        localStorage.removeItem('scribeai_pin_hash');
        localStorage.removeItem('selectedMode');
        localStorage.removeItem('customPromptInstructions');
        localStorage.removeItem('selectedTimezone');
        localStorage.removeItem('autoDownloadEnabled');
        localStorage.removeItem('gemini_api_key');
        localStorage.removeItem('scribeai_refresh_count');
        
        this.clearPinInputs();
        this.showSetPinScreen();
    }
  }

  private clearPinInputs(): void {
    this.pinInputs.forEach(input => (input.value = ''));
    this.pinInputs[0].focus();
    this.pinSubmitButton.disabled = true;
  }
  
  private unlockApp(): void {
    this.pinModal.style.display = 'none';
    this.initializeApp();
  }

  private bindEventListeners(): void {
    // Desktop buttons
    this.fabRecord.addEventListener('click', () => this.startFullRecordingSession());
    this.newButton.addEventListener('click', () => this.createNewNote());
    this.uploadButton.addEventListener('click', () => this.triggerFileUpload());
    this.downloadAudioButton.addEventListener('click', () => this.downloadFullAudio());
    this.downloadNoteButton.addEventListener('click', () => this.downloadPolishedNote());
    this.themeToggleButton.addEventListener('click', () => this.toggleTheme());
    this.copyButton.addEventListener('click', () => this.copyPolishedNote());
    this.copyMetaButton.addEventListener('click', () => this.copyMetadata());
    this.settingsButton.addEventListener('click', (e) => {
        e.stopPropagation();
        this.toggleSettingsMenu();
    });
    
    // Mobile buttons
    this.bottomNavRecord.addEventListener('click', () => this.startFullRecordingSession());
    this.bottomNavNew.addEventListener('click', () => this.createNewNote());
    this.bottomNavUpload.addEventListener('click', () => this.triggerFileUpload());
    this.moreMenuButton.addEventListener('click', (e) => {
      e.stopPropagation();
      this.toggleMoreMenu();
    });

    // Shared buttons
    this.stopButton.addEventListener('click', () => this.stopFullRecordingSession());
    this.pauseButton.addEventListener('click', () => this.handlePauseResume());
    this.lapButton.addEventListener('click', () => this.handleLap());
    this.audioUploadInput.addEventListener('change', (e) => this.handleFileUpload(e));
    this.copyRawButton.addEventListener('click', () => this.copyRawTranscription());
    
    this.editCustomPromptButton.addEventListener('click', () => this.openCustomPromptModal());
    this.saveCustomPromptButton.addEventListener('click', () => this.saveCustomPrompt());
    this.cancelCustomPromptButton.addEventListener('click', () => this.closeCustomPromptModal());
    this.infoModalCloseButton.addEventListener('click', () => this.closeInfoModal());
    this.timezoneModalCloseButton.addEventListener('click', () => this.closeTimezoneModal());

    this.modeSelectorButton.addEventListener('click', (e) => {
        e.stopPropagation();
        this.toggleModeList();
    });

    this.timezoneSelectorButton.addEventListener('click', (e) => {
      e.stopPropagation();
      this.toggleTimezoneList();
    });

    this.autoDownloadToggle.addEventListener('change', () => this.handleAutoDownloadToggle());

    // API Key Listeners
    this.saveApiKeyButton.addEventListener('click', () => this.handleSaveApiKey());
    this.updateApiKeyButton.addEventListener('click', () => this.handleUpdateApiKey());
    this.toggleApiKeyVisibilityButton.addEventListener('click', () => this.toggleApiKeyVisibility());

    document.addEventListener('click', (e) => this.handleDocumentClick(e));
    window.addEventListener('resize', this.handleResize.bind(this));
  }

  private initAutoDownload(): void {
    const savedState = localStorage.getItem('autoDownloadEnabled');
    this.autoDownloadEnabled = savedState === 'true';
    this.autoDownloadToggle.checked = this.autoDownloadEnabled;
  }
  
  private handleAutoDownloadToggle(): void {
      this.autoDownloadEnabled = this.autoDownloadToggle.checked;
      localStorage.setItem('autoDownloadEnabled', String(this.autoDownloadEnabled));
  }
  
  private initTabs(): void {
    this.tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const targetId = button.dataset.tabTarget;
            if (!targetId) return;

            this.tabButtons.forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');
            
            this.updateTabIndicator();

            this.tabPanes.forEach(pane => {
                pane.classList.remove('active');
                if (`#${pane.id}` === targetId) {
                    pane.classList.add('active');
                }
            });
        });
    });
    this.updateTabIndicator();
  }

  private updateTabIndicator(): void {
    const activeButton = document.querySelector('.tab-button.active') as HTMLButtonElement;
    if (activeButton && this.tabIndicator) {
      this.tabIndicator.style.width = `${activeButton.offsetWidth}px`;
      this.tabIndicator.style.left = `${activeButton.offsetLeft}px`;
    }
  }

  private initCustomModeSelector(): void {
    this.modeList.innerHTML = ''; // Clear existing
    for (const key in MODES) {
        const mode = MODES[key as ModeID];
        
        const wrapper = document.createElement('div');
        wrapper.className = 'mode-option-wrapper';

        const optionButton = document.createElement('button');
        optionButton.className = 'mode-option';
        optionButton.textContent = mode.name;
        optionButton.dataset.modeId = mode.id;

        optionButton.addEventListener('click', () => {
            this.handleModeChange(mode.id);
            this.closeModeList();
        });
        
        const infoButton = document.createElement('button');
        infoButton.className = 'mode-info-button';
        infoButton.innerHTML = '<i class="fas fa-info-circle"></i>';
        infoButton.title = `About ${mode.name} mode`;
        infoButton.addEventListener('click', (e) => {
            e.stopPropagation();
            this.showInfoModal(mode);
        });
        
        wrapper.appendChild(optionButton);
        wrapper.appendChild(infoButton);
        this.modeList.appendChild(wrapper);
    }
  }

  private initTimezoneSelector(): void {
    const createTimezoneOption = (tz: string, isMobile: boolean) => {
      const optionButton = document.createElement('button');
      optionButton.className = 'mode-option';
      optionButton.textContent = tz;
      optionButton.dataset.tz = tz;
      optionButton.addEventListener('click', () => {
        this.handleTimezoneChange(tz);
        if(isMobile) {
            this.closeTimezoneModal();
        } else {
            this.closeTimezoneList();
        }
      });
      return optionButton;
    };
    
    this.timezoneList.innerHTML = ''; // Clear existing for desktop
    this.mobileTimezoneList.innerHTML = ''; // Clear for mobile
    
    TIMEZONES.forEach(tz => {
        this.timezoneList.appendChild(createTimezoneOption(tz, false));
        this.mobileTimezoneList.appendChild(createTimezoneOption(tz, true));
    });
    
    const savedTimezone = localStorage.getItem('selectedTimezone');
    if (savedTimezone && TIMEZONES.includes(savedTimezone)) {
        this.currentTimezone = savedTimezone;
    } else {
        this.currentTimezone = 'Aurangabad'; // Default
    }
    this.updateTimezoneDisplay();
  }
  
  private getIanaTimezone(name: string): string {
    switch (name) {
        case 'Aurangabad':
        case 'Pune':
        case 'Mumbai':
            return 'Asia/Kolkata';
        case 'Warsaw':
            return 'Europe/Warsaw';
        case 'UTC':
            return 'UTC';
        default:
            return name; // Fallback
    }
  }

  private handleTimezoneChange(newTimezone: string): void {
    this.currentTimezone = newTimezone;
    localStorage.setItem('selectedTimezone', this.currentTimezone);
    this.updateTimezoneDisplay();
    this.updateMetadataDisplay();
    this.initMoreMenu(); // Re-init to update text
  }
  
  private updateTimezoneDisplay(): void {
    if (this.currentTimezoneNameSpan) {
        this.currentTimezoneNameSpan.textContent = this.currentTimezone;
    }
    document.querySelectorAll('.mode-option[data-tz]').forEach(opt => {
        const button = opt as HTMLButtonElement;
        if (button.dataset.tz === this.currentTimezone) {
            button.classList.add('active');
        } else {
            button.classList.remove('active');
        }
    });
  }

  private toggleTimezoneList(): void {
    if (this.timezoneList.classList.contains('show')) {
        this.closeTimezoneList();
    } else {
        this.timezoneList.classList.add('show');
        this.timezoneSelectorButton.classList.add('open');
    }
  }

  private closeTimezoneList(): void {
      this.timezoneList.classList.remove('show');
      this.timezoneSelectorButton.classList.remove('open');
  }

  private toggleModeList(): void {
    if (this.modeList.classList.contains('show')) {
        this.closeModeList();
    } else {
        this.closeSettingsMenu();
        this.closeMoreMenu();
        this.modeList.classList.add('show');
        this.modeSelectorButton.classList.add('open');
    }
  }

  private closeModeList(): void {
      this.modeList.classList.remove('show');
      this.modeSelectorButton.classList.remove('open');
  }
  
  private toggleSettingsMenu(): void {
    if (this.settingsMenuList.classList.contains('show')) {
        this.closeSettingsMenu();
    } else {
        this.closeModeList();
        this.closeMoreMenu();
        this.settingsMenuList.classList.add('show');
    }
  }
  
  private closeSettingsMenu(): void {
      this.settingsMenuList.classList.remove('show');
      this.closeTimezoneList();
  }

  private handleDocumentClick(event: MouseEvent): void {
      if (!this.modeSelectorContainer.contains(event.target as Node)) {
          this.closeModeList();
      }
      if (!this.settingsMenuContainer.contains(event.target as Node)) {
          this.closeSettingsMenu();
      } else {
          // If click is inside settings menu, but outside timezone selector, close timezone list.
          if (!this.timezoneSelectorContainer.contains(event.target as Node)) {
              this.closeTimezoneList();
          }
      }
      if (!this.moreMenuContainer.contains(event.target as Node)) {
          this.closeMoreMenu();
      }
  }

  private showInfoModal(mode: Mode): void {
    this.infoModalTitle.textContent = mode.name;
    this.infoModalContent.innerHTML = `<pre>${mode.instructions}</pre>`;
    this.infoModal.style.display = 'flex';
  }

  private closeInfoModal(): void {
      this.infoModal.style.display = 'none';
  }
  
  private openTimezoneModal(): void {
    this.timezoneModal.style.display = 'flex';
  }

  private closeTimezoneModal(): void {
    this.timezoneModal.style.display = 'none';
  }

  private loadAndSetInitialMode(): void {
    const savedMode = localStorage.getItem('selectedMode') as ModeID;
    if (savedMode && MODES[savedMode]) {
      this.currentModeId = savedMode;
    } else {
      this.currentModeId = 'doctor'; // Default mode
    }
    this.updateModeDisplay();
    this.updateCustomPromptButtonVisibility();
  }
  
  private updateModeDisplay(): void {
    const currentMode = MODES[this.currentModeId];
    if (currentMode) {
        this.currentModeNameSpan.textContent = currentMode.name;
    }

    this.modeList.querySelectorAll('.mode-option-wrapper').forEach(opt => {
        const wrapper = opt as HTMLDivElement;
        const button = wrapper.querySelector('.mode-option') as HTMLButtonElement;
        if (button && button.dataset.modeId === this.currentModeId) {
            wrapper.classList.add('active');
        } else {
            wrapper.classList.remove('active');
        }
    });
  }

  private handleModeChange(newModeId: ModeID): void {
    this.currentModeId = newModeId;
    localStorage.setItem('selectedMode', this.currentModeId);
    this.updateModeDisplay();
    if(this.currentNote) {
        this.currentNote.modeId = newModeId;
        this.updateMetadataDisplay();
    }
    this.updateCustomPromptButtonVisibility();
    this.initMoreMenu(); // Re-init to show/hide custom prompt option
  }

  private loadCustomPrompt(): void {
    const savedPrompt = localStorage.getItem('customPromptInstructions');
    // A helpful default prompt for first-time users.
    this.customPromptInstructions = savedPrompt || `You are a helpful assistant. Please follow these instructions:
- Summarize the text into three bullet points.
- Identify any questions asked within the text.
- List all action items clearly using markdown checkboxes.`;
    MODES.custom.instructions = this.customPromptInstructions;
  }

  private updateCustomPromptButtonVisibility(): void {
    if (this.currentModeId === 'custom') {
        this.customPromptSettingsItem.style.display = 'block';
    } else {
        this.customPromptSettingsItem.style.display = 'none';
    }
  }

  private openCustomPromptModal(): void {
    this.customPromptTextarea.value = this.customPromptInstructions;
    this.customPromptModal.style.display = 'flex';
    this.customPromptTextarea.focus();
  }

  private closeCustomPromptModal(): void {
    this.customPromptModal.style.display = 'none';
  }

  private saveCustomPrompt(): void {
    const newPrompt = this.customPromptTextarea.value.trim();
    if (newPrompt) {
      this.customPromptInstructions = newPrompt;
      MODES.custom.instructions = newPrompt;
      localStorage.setItem('customPromptInstructions', newPrompt);
      this.closeCustomPromptModal();
    } else {
      // Optional: show an error message that prompt can't be empty
      this.customPromptTextarea.placeholder = 'Prompt cannot be empty. Please enter your instructions.';
    }
  }

  private handleResize(): void {
    this.updateTabIndicator();
    if (this.isRecording) {
      this.waveformVisualizer?.resize();
    }
  }

  private initTheme(): void {
    const savedTheme = localStorage.getItem('theme');
    const isLight = savedTheme === 'light';
    
    if (isLight) {
        document.body.classList.add('light-mode');
    } else {
        document.body.classList.remove('light-mode');
    }

    this.updateThemeIcons(isLight);
    this.updateThemeColorMeta();
  }

  private toggleTheme(): void {
    const isLight = document.body.classList.toggle('light-mode');
    localStorage.setItem('theme', isLight ? 'light' : 'dark');
    this.updateThemeIcons(isLight);
    this.updateThemeColorMeta();
    this.waveformVisualizer?.updateConfig({ 
        waveColor: getComputedStyle(document.documentElement).getPropertyValue('--color-primary').trim() 
    });
  }
  
  private updateThemeIcons(isLight: boolean): void {
    const iconClassAdd = isLight ? 'fa-moon' : 'fa-sun';
    const iconClassRemove = isLight ? 'fa-sun' : 'fa-moon';

    this.themeToggleIcon.classList.remove(iconClassRemove);
    this.themeToggleIcon.classList.add(iconClassAdd);

    if (this.moreMenuThemeToggleIcon) {
        this.moreMenuThemeToggleIcon.classList.remove(iconClassRemove);
        this.moreMenuThemeToggleIcon.classList.add(iconClassAdd);
    }
  }

  private updateThemeColorMeta(): void {
    const themeColor = getComputedStyle(document.body).getPropertyValue('--color-surface').trim();
    const metaThemeColor = document.querySelector('meta[name="theme-color"]');
    if (metaThemeColor) {
      metaThemeColor.setAttribute('content', themeColor);
    }
  }
  
  private handleAutoDownloadFromMenu(): void {
    this.autoDownloadEnabled = !this.autoDownloadEnabled;
    this.autoDownloadToggle.checked = this.autoDownloadEnabled;
    localStorage.setItem('autoDownloadEnabled', String(this.autoDownloadEnabled));
    this.initMoreMenu(); // Re-init to update the checkmark
  }

  private initMoreMenu(): void {
    this.moreMenuList.innerHTML = '';
    
    const actions: ({
        id: string;
        icon: string;
        text: string;
        action: () => void;
        state?: string;
        condition?: boolean;
    })[] = [
        { id: 'copy', icon: 'fa-copy', text: 'Copy Polished Note', action: () => this.copyPolishedNote(), condition: true },
        { id: 'downloadAudio', icon: 'fa-file-audio', text: 'Download Audio', action: () => this.downloadFullAudio(), condition: true },
        { id: 'download', icon: 'fa-download', text: 'Download Note', action: () => this.downloadPolishedNote(), condition: true },
        { id: 'copyMeta', icon: 'fa-clipboard', text: 'Copy Metadata', action: () => this.copyMetadata(), condition: true },
        { id: 'updateKey', icon: 'fa-key', text: 'Update API Key', action: () => this.handleUpdateApiKey(), condition: true },
        { id: 'timezone', icon: 'fa-globe-americas', text: 'Timezone', action: () => this.openTimezoneModal(), state: this.currentTimezone, condition: true },
        { id: 'autoDownload', icon: 'fa-file-download', text: 'Auto-download Note', action: () => this.handleAutoDownloadFromMenu(), state: this.autoDownloadEnabled ? 'On' : 'Off', condition: true },
        { id: 'editCustom', icon: 'fa-pencil-alt', text: 'Edit Custom Prompt', action: () => this.openCustomPromptModal(), condition: this.currentModeId === 'custom' },
        { id: 'theme', icon: 'fa-sun', text: 'Toggle Theme', action: () => this.toggleTheme(), condition: true }
    ];

    actions.forEach(action => {
      if (!action.condition) return;

      const button = document.createElement('button');
      button.className = 'more-menu-item';
      button.title = action.text;
      button.addEventListener('click', () => {
          action.action();
          this.closeMoreMenu();
      });

      const icon = document.createElement('i');
      icon.className = `fas ${action.icon}`;
      if (action.id === 'theme') {
          this.moreMenuThemeToggleIcon = icon;
      }

      const textSpan = document.createElement('span');
      textSpan.className = 'menu-item-text';
      textSpan.textContent = action.text;

      button.appendChild(icon);
      button.appendChild(textSpan);
      
      if(action.state !== undefined) {
        const stateSpan = document.createElement('span');
        stateSpan.className = 'menu-item-state';
        stateSpan.textContent = action.state;
        button.appendChild(stateSpan);
      }

      this.moreMenuList.appendChild(button);
    });
    this.initTheme(); // Ensure mobile theme icon is correct
  }

  private toggleMoreMenu(): void {
    this.closeModeList();
    this.closeSettingsMenu();
    this.moreMenuList.classList.toggle('show');
  }

  private closeMoreMenu(): void {
      this.moreMenuList.classList.remove('show');
  }

  private setGlobalStatus(text: string, isProcessing = false, isError = false): void {
    if (!this.globalStatus) return;
    
    this.globalStatus.classList.remove('processing', 'error');
    
    let iconHtml = '';
    if (isProcessing) {
        this.globalStatus.classList.add('processing');
        iconHtml = '<i class="fas fa-spinner fa-spin"></i>';
    } else if (isError) {
        this.globalStatus.classList.add('error');
    }

    const textSpan = document.createElement('span');
    textSpan.textContent = text;
    
    this.globalStatus.innerHTML = iconHtml;
    this.globalStatus.appendChild(textSpan);
  }

  private async startFullRecordingSession(): Promise<void> {
    if (this.isRecording || this.isProcessing) return;

    this.isRecording = true;
    this.isPaused = false;
    this.lapCount = 0;
    this.allRawLapText = '';
    this.totalDurationMs = 0;
    this.sessionAudioChunks = [];
    this.sessionMimeType = ''; // Reset for new session
    
    if (this.currentNote) {
      this.currentNote.timestamp = Date.now();
      this.currentNote.duration = 0;
      this.currentNote.audioSize = 0;
      this.currentNote.promptTokens = 0;
      this.currentNote.completionTokens = 0;
      this.currentNote.cost = 0;
    }
    
    const rawPlaceholder = this.rawTranscription.getAttribute('placeholder') || '';
    this.rawTranscription.textContent = rawPlaceholder;
    this.rawTranscription.classList.add('placeholder-active');
    
    const polishedPlaceholder = this.polishedNote.getAttribute('placeholder') || '';
    this.polishedNote.innerHTML = polishedPlaceholder;
    this.polishedNote.classList.add('placeholder-active');

    this.updateMetadataDisplay();
    this.showRecordingDialog();
    await this._startNextRecordingSegment();
  }

  private async stopFullRecordingSession(): Promise<void> {
    if (!this.isRecording || this.isProcessing) return;
    this.stopReason = 'stop';
    this.mediaRecorder?.stop();
  }

  private async handleLap(): Promise<void> {
    if (!this.isRecording || this.isPaused || this.isProcessing) return;
    this.stopReason = 'lap';
    this.mediaRecorder?.stop();
  }

  private async handlePauseResume(): Promise<void> {
    if (!this.isRecording || this.isProcessing) return;

    const icon = this.pauseButton.querySelector('i');
    if (!icon) return;

    if (this.isPaused) { // RESUMING
      this.mediaRecorder?.resume();
      this.isPaused = false;
      this.waveformVisualizer?.resume();
      icon.classList.remove('fa-play');
      icon.classList.add('fa-pause');
      this.liveRecordingTitle.textContent = 'Recording...';
      
      // Reset the start time for the current part of the segment
      this.recordingStartTime = Date.now();
      
      if (this.timerIntervalId) clearInterval(this.timerIntervalId);
      this.timerIntervalId = window.setInterval(() => this.updateLiveTimer(), 50);
    } else { // PAUSING
      this.mediaRecorder?.pause();
      this.isPaused = true;
      this.waveformVisualizer?.pause();
      
      // Capture the elapsed time for this part of the segment and add it to the total
      const segmentPartDuration = Date.now() - this.recordingStartTime;
      this.totalDurationMs += segmentPartDuration;

      icon.classList.remove('fa-pause');
      icon.classList.add('fa-play');
      this.liveRecordingTitle.textContent = 'Paused';
      if (this.timerIntervalId) clearInterval(this.timerIntervalId); // Stop timer
    }
  }

  private updateLiveTimer(): void {
    if (!this.isRecording || !this.liveRecordingTimerDisplay || this.isPaused) return;
    const now = Date.now();
    const elapsedMs = (now - this.recordingStartTime) + this.totalDurationMs;

    const totalSeconds = Math.floor(elapsedMs / 1000);
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    const hundredths = Math.floor((elapsedMs % 1000) / 10);

    this.liveRecordingTimerDisplay.textContent = `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}.${String(hundredths).padStart(2, '0')}`;
  }

  private showRecordingDialog(): void {
    this.fabRecord.style.display = 'none';
    this.bottomNav.style.display = 'none';
    this.recordingDialog.classList.add('show');
    
    this.liveRecordingTitle.textContent = 'Recording...';
    
    this.recordingStartTime = Date.now();
    this.updateLiveTimer();
    if (this.timerIntervalId) clearInterval(this.timerIntervalId);
    this.timerIntervalId = window.setInterval(() => this.updateLiveTimer(), 50);
  }

  private hideRecordingDialog(): void {
    if (window.matchMedia("(max-width: 767px)").matches) {
      this.bottomNav.style.display = 'flex';
      this.fabRecord.style.display = 'none';
    } else {
      this.bottomNav.style.display = 'none';
      this.fabRecord.style.display = 'flex';
    }
    
    this.recordingDialog.classList.remove('show');
    
    this.waveformVisualizer?.stop();
    this.waveformVisualizer = null;

    if (this.timerIntervalId) {
      clearInterval(this.timerIntervalId);
      this.timerIntervalId = null;
    }
  }

  private async _startNextRecordingSegment(): Promise<void> {
    try {
      this.audioChunks = [];
      if (!this.stream) {
          try {
              this.stream = await navigator.mediaDevices.getUserMedia({audio: true});
          } catch (err) {
              console.error('Failed with basic constraints:', err);
              this.stream = await navigator.mediaDevices.getUserMedia({
                  audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
              });
          }
      }

      if (this.stream && !this.waveformVisualizer) {
        this.waveformVisualizer = new WaveformVisualizer(this.liveWaveformCanvas, {
            waveColor: getComputedStyle(document.documentElement).getPropertyValue('--color-primary').trim(),
            smoothingTimeConstant: 0.6,
            lineWidth: 2,
            fftSize: 2048,
        });
        await this.waveformVisualizer.start(this.stream);
      }
      
      // Determine and set the MIME type for the entire session if not already set
      if (!this.sessionMimeType) {
        const mimeTypes = [
            'audio/mp4', // Prioritize for broader compatibility
            'audio/webm;codecs=opus',
            'audio/webm',
        ];
        this.sessionMimeType = mimeTypes.find(type => MediaRecorder.isTypeSupported(type)) || 'audio/webm'; // Default fallback
      }
      
      this.mediaRecorder = new MediaRecorder(this.stream, { mimeType: this.sessionMimeType });

      this.mediaRecorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          this.audioChunks.push(event.data);
          this.sessionAudioChunks.push(event.data);
          // Enable download button once we have audio data
          if (this.downloadAudioButton.disabled) {
            this.downloadAudioButton.disabled = false;
          }
        }
      };

      this.mediaRecorder.onstop = async () => {
        // If the recorder was paused, the pre-pause duration is already in totalDurationMs.
        // We only need to add the duration of the final active segment part.
        if (!this.isPaused) {
            const segmentDuration = Date.now() - this.recordingStartTime;
            this.totalDurationMs += segmentDuration;
        }
        
        if (this.audioChunks.length > 0) {
            const audioBlob = new Blob(this.audioChunks, { type: this.sessionMimeType });
            if(this.currentNote) {
                this.currentNote.audioSize += audioBlob.size;
            }
            await this.processAudioSegment(audioBlob);
        } else { // No audio captured in this segment
            if (this.stopReason === 'lap') {
                await this._startNextRecordingSegment(); // Just restart for next lap
            } else { // Final stop with no final audio
                this.setGlobalStatus('Polishing note...', true);
                await this.getPolishedNote();
                this.resetToIdleState();
            }
        }
      };

      this.mediaRecorder.start();
      this.recordingStartTime = Date.now();
      this.liveRecordingTitle.textContent = 'Recording...';
      if (this.timerIntervalId) clearInterval(this.timerIntervalId);
      this.timerIntervalId = window.setInterval(() => this.updateLiveTimer(), 50);

    } catch (error) {
      console.error('Error starting recording:', error);
      const message = error instanceof Error ? error.message : "Unknown error";
      this.setGlobalStatus(`Error: ${message}`, false, true);
      this.resetToIdleState();
    }
  }

  private async processAudioSegment(audioBlob: Blob): Promise<void> {
    if (this.isProcessing) return;
    this.isProcessing = true;
    this.setLiveControls(false); // Disable buttons
    this.lapCount++;
    this.liveRecordingTitle.textContent = `Processing Lap ${this.lapCount}...`;
    if (this.timerIntervalId) clearInterval(this.timerIntervalId);

    try {
      const base64Audio = await this.blobToBase64(audioBlob);
      if (!base64Audio) throw new Error('Failed to convert audio');

      const mimeType = this.sessionMimeType || 'audio/webm';
      const transcriptionText = await this.getTranscription(base64Audio, mimeType);
      
      const segmentStartTime = this.formatDuration(this.totalDurationMs - (Date.now() - this.recordingStartTime));
      const segmentEndTime = this.formatDuration(this.totalDurationMs);
      const lapHeader = `\n\n--- LAP ${this.lapCount} (${segmentStartTime} - ${segmentEndTime}) ---\n\n`;
      this.allRawLapText += lapHeader + (transcriptionText || '[No speech detected]');
      
      this.rawTranscription.textContent = this.allRawLapText;
      if (this.rawTranscription.classList.contains('placeholder-active')) {
          this.rawTranscription.classList.remove('placeholder-active');
      }
      if(this.currentNote) this.currentNote.rawTranscription = this.allRawLapText;

    } catch (error) {
        console.error('Error processing audio segment:', error);
        this.setGlobalStatus('Error processing segment.', false, true);
    } finally {
        if (this.stopReason === 'lap') {
            await this._startNextRecordingSegment();
        } else { // 'stop'
            this.liveRecordingTitle.textContent = 'Polishing final note...';
            await this.getPolishedNote();
            this.resetToIdleState();
        }
        this.isProcessing = false;
        this.setLiveControls(true); // Re-enable buttons if continuing
    }
  }

  private triggerFileUpload(): void {
    if (this.isRecording || this.isProcessing) return;
    this.audioUploadInput.click();
  }

  private async handleFileUpload(event: Event): Promise<void> {
    const input = event.target as HTMLInputElement;
    if (!input.files || input.files.length === 0) {
        return;
    }
    const file = input.files[0];

    // Enhanced validation for audio files
    const allowedTypes = [
        'audio/',           // All audio MIME types (audio/mp3, audio/wav, etc.)
        'video/mp4',        // MP4 files (often contain audio-only content)
        'video/quicktime',  // MOV files (can contain audio)
        'application/octet-stream' // Files with unknown/missing MIME types
    ];

    const isValidMimeType = allowedTypes.some(type => 
        type.endsWith('/') ? file.type.startsWith(type) : file.type === type
    );

    // Fallback: check file extension for files with missing/incorrect MIME types
    const allowedExtensions = ['.mp3', '.wav', '.aac', '.mp4', '.m4a', '.ogg', '.flac', '.wma'];
    const fileExtension = file.name.toLowerCase().substring(file.name.lastIndexOf('.'));
    const isValidExtension = allowedExtensions.includes(fileExtension);

    if (!isValidMimeType && !isValidExtension) {
        this.setGlobalStatus('Error: Invalid file type. Please upload an audio file.', false, true);
        input.value = ''; // Reset for next selection
        setTimeout(() => {
            if (this.globalStatus.textContent?.includes('Invalid file type')) {
                this.setGlobalStatus('Ready to record');
            }
        }, 3000);
        return;
    }
    
    if (this.isRecording || this.isProcessing) {
        this.setGlobalStatus('Please wait for the current process to finish.');
        input.value = '';
        return;
    }
    
    this.createNewNote();
    
    this.isProcessing = true;
    this.fabRecord.disabled = true;

    try {
        if (this.currentNote) {
            this.currentNote.audioSize = file.size;
            this.currentNote.duration = 0; // Duration is not available for uploads
            this.totalDurationMs = 0;
        }
        this.updateMetadataDisplay();

        this.setGlobalStatus(`Processing ${file.name}...`, true);

        const base64Audio = await this.blobToBase64(file);
        if (!base64Audio) throw new Error('Failed to read the audio file.');

        // Enhanced logging for transcription debugging
        console.log('File upload debug info:', {
            fileName: file.name,
            fileSize: file.size,
            mimeType: file.type,
            base64Length: base64Audio.length,
            fileExtension: file.name.toLowerCase().substring(file.name.lastIndexOf('.'))
        });

        // Normalize MIME type for transcription service
        let normalizedMimeType = file.type;
        if (file.type === 'video/mp4') {
            normalizedMimeType = 'audio/mp4'; // Many transcription services expect audio/ prefix
            console.log('Normalized MIME type from video/mp4 to audio/mp4');
        } else if (file.type === 'application/octet-stream') {
            // Guess MIME type from file extension for unknown types
            const ext = file.name.toLowerCase().substring(file.name.lastIndexOf('.'));
            const mimeMap: { [key: string]: string } = {
                '.mp3': 'audio/mpeg',
                '.wav': 'audio/wav',
                '.aac': 'audio/aac',
                '.m4a': 'audio/mp4',
                '.ogg': 'audio/ogg',
                '.flac': 'audio/flac'
            };
            normalizedMimeType = mimeMap[ext] || file.type;
            console.log(`Guessed MIME type for ${ext}: ${normalizedMimeType}`);
        }

        console.log('Sending to transcription service with MIME type:', normalizedMimeType);

        let transcriptionText: string;
        try {
            transcriptionText = await this.getTranscription(base64Audio, normalizedMimeType, true);
            console.log('Transcription successful, length:', transcriptionText?.length || 0);
        } catch (transcriptionError) {
            console.error('Detailed transcription error:', {
                error: transcriptionError,
                errorMessage: transcriptionError instanceof Error ? transcriptionError.message : 'Unknown error',
                errorStack: transcriptionError instanceof Error ? transcriptionError.stack : undefined,
                fileInfo: {
                    name: file.name,
                    size: file.size,
                    type: file.type,
                    normalizedType: normalizedMimeType
                }
            });
            throw transcriptionError; // Re-throw to maintain existing error handling
        }
        
        this.allRawLapText = transcriptionText || '[No speech detected]';
        this.rawTranscription.textContent = this.allRawLapText;
        if (this.rawTranscription.classList.contains('placeholder-active')) {
            this.rawTranscription.classList.remove('placeholder-active');
        }
        if(this.currentNote) this.currentNote.rawTranscription = this.allRawLapText;
        
        await this.getPolishedNote();
    } catch (error) {
        console.error('Error processing uploaded file:', error);
        const errorMessage = error instanceof Error ? error.message : "Upload failed";
        this.setGlobalStatus(`Error: ${errorMessage}`, false, true);
    } finally {
        this.isProcessing = false;
        this.fabRecord.disabled = false;
        input.value = ''; // Reset for next selection
        this.updateMetadataDisplay();
    }
}

  private async blobToBase64(blob: Blob): Promise<string> {
      const reader = new FileReader();
      const readResult = new Promise<string>((resolve, reject) => {
        reader.onloadend = () => {
          try {
            resolve((reader.result as string).split(',')[1]);
          } catch (err) {
            reject(err);
          }
        };
        reader.onerror = () => reject(reader.error);
      });
      reader.readAsDataURL(blob);
      return readResult;
  }

  private async getTranscription(base64Audio: string, mimeType: string, isUpload: boolean = false): Promise<string> {
    const context = isUpload ? 'file' : `Lap ${this.lapCount}`;
    try {
      this.setGlobalStatus(`Transcribing ${context}...`, true);
      const contents = {
          parts: [
            {text: 'Transcribe this audio with the following format:\n\n[TIMESTAMP] SPEAKER: exact spoken words\n\nInclude timestamps every 10-15 seconds, detect different speakers (Speaker 1, Speaker 2, etc.), mark pauses with [PAUSE], unclear words with [UNCLEAR], and background sounds with [BACKGROUND: description]. Capture everything exactly as spoken including filler words, repetitions, and false starts.'},
            {inlineData: {mimeType: mimeType, data: base64Audio}},
          ],
      };
      const response = await this.genAI.models.generateContent({ model: MODEL_NAME, contents: contents });
      
      if (response.usageMetadata && this.currentNote) {
        this.currentNote.promptTokens += response.usageMetadata.promptTokenCount ?? 0;
        this.currentNote.completionTokens += response.usageMetadata.candidatesTokenCount ?? 0;
        this.updateNoteCost();
        this.updateMetadataDisplay();
      }
      return response.text;
    } catch (error) {
      console.error(`Error getting transcription for ${context}:`, error);
      const message = error instanceof Error ? error.message : String(error);
      if (message.includes('API key not valid')) {
        this.setGlobalStatus('API Key is invalid. Please update it.', false, true);
        this.handleUpdateApiKey();
        this.disableAppFeatures();
      } else {
        this.setGlobalStatus(`Error transcribing ${context}.`, false, true);
      }
      return `[Error during transcription: ${message}]`;
    }
  }

  private async getPolishedNote(): Promise<void> {
    try {
      if (!this.allRawLapText.trim()) {
        this.setGlobalStatus('No transcription to polish');
        this.polishedNote.innerHTML = '<p><em>No transcription available to polish.</em></p>';
        this.polishedNote.classList.add('placeholder-active');
        return;
      }
      this.setGlobalStatus('Polishing note...', true);
      const mode = MODES[this.currentModeId] || MODES.journal;
      const ianaTimezone = this.getIanaTimezone(this.currentTimezone);
      const location = this.currentTimezone;
      const noteTimestamp = this.currentNote ? this.currentNote.timestamp : Date.now();
      const timestamp = new Date(noteTimestamp).toLocaleString('en-US', {
          timeZone: ianaTimezone, dateStyle: 'full', timeStyle: 'short',
      });
      const prompt = `You are a specialized AI assistant that transforms raw audio transcription into a specific, structured format based on the user's selected 'mode'.

Your task is to follow the instructions for the selected mode precisely and generate a markdown response.
The note MUST begin with the provided location and timestamp.
Do not add any commentary before or after the markdown content.

Location: ${location}
Timestamp: ${timestamp}
Mode: ${mode.name}
Instructions:
${mode.instructions}

---

Raw transcription (from multiple laps):
${this.allRawLapText}`;
      
      const response = await this.genAI.models.generateContent({ model: MODEL_NAME, contents: prompt });

      if (response.usageMetadata && this.currentNote) {
        this.currentNote.promptTokens += response.usageMetadata.promptTokenCount ?? 0;
        this.currentNote.completionTokens += response.usageMetadata.candidatesTokenCount ?? 0;
        this.updateNoteCost();
      }
      const polishedText = response.text;
      if (polishedText) {
        const htmlContent = await marked.parse(String(polishedText));
        this.polishedNote.innerHTML = htmlContent;
        this.polishedNote.classList.remove('placeholder-active');
        if (this.currentNote) this.currentNote.polishedNote = polishedText;
        this.setGlobalStatus('Note polished.');
        if (this.autoDownloadEnabled) {
          // Add a small delay so the user can see the status change before download
          setTimeout(() => this.downloadPolishedNote(), 500);
        }
      } else {
        this.setGlobalStatus('Polishing failed or returned empty.', false, true);
        this.polishedNote.innerHTML = '<p><em>Polishing returned empty. Raw transcription is available.</em></p>';
        this.polishedNote.classList.add('placeholder-active');
      }
    } catch (error) {
      console.error('Error polishing note:', error);
      const message = error instanceof Error ? error.message : String(error);
      if (message.includes('API key not valid')) {
        this.setGlobalStatus('API Key is invalid. Please update it.', false, true);
        this.handleUpdateApiKey();
        this.disableAppFeatures();
      } else {
        this.setGlobalStatus('Error polishing note. Please try again.', false, true);
      }
      this.polishedNote.innerHTML = `<p><em>Error during polishing: ${message}</em></p>`;
      this.polishedNote.classList.add('placeholder-active');
    } finally {
        this.updateMetadataDisplay();
    }
  }

  private setButtonState(button: HTMLButtonElement, state: 'success' | 'error'): void {
    const icon = button.querySelector('i');
    if (!icon) return;

    const originalIconClasses = button.dataset.originalIcon || icon.className;
    if (!button.dataset.originalIcon) {
        button.dataset.originalIcon = originalIconClasses;
    }

    button.classList.remove('copied', 'error'); // remove previous states
    button.classList.add(state === 'success' ? 'copied' : 'error');
    icon.className = `fas ${state === 'success' ? 'fa-check' : 'fa-times'}`;

    const existingTimeoutId = parseInt(button.dataset.timeoutId || '0', 10);
    if (existingTimeoutId) {
        clearTimeout(existingTimeoutId);
    }

    const timeoutId = window.setTimeout(() => {
        button.classList.remove('copied', 'error');
        icon.className = originalIconClasses;
        delete button.dataset.timeoutId;
        delete button.dataset.originalIcon;
    }, 2000);
    button.dataset.timeoutId = String(timeoutId);
  }

  private async copyPolishedNote(): Promise<void> {
    if (this.polishedNote.classList.contains('placeholder-active') || this.polishedNote.innerText.trim() === '') {
      console.warn('No polished note content to copy.');
      return;
    }
    
    try {
      const htmlBlob = new Blob([this.polishedNote.innerHTML], { type: 'text/html' });
      const textBlob = new Blob([this.polishedNote.innerText], { type: 'text/plain' });
      const item = new ClipboardItem({ 'text/html': htmlBlob, 'text/plain': textBlob });
      await navigator.clipboard.write([item]);
      this.setButtonState(this.copyButton, 'success');
    } catch (err) {
      console.error('Failed to copy rich text, falling back to plain text: ', err);
      try {
        await navigator.clipboard.writeText(this.polishedNote.innerText);
        this.setButtonState(this.copyButton, 'success');
      } catch (fallbackErr) {
        console.error('Failed to copy as plain text: ', fallbackErr);
        this.setButtonState(this.copyButton, 'error');
      }
    }
  }

  private async copyRawTranscription(): Promise<void> {
    const rawText = this.rawTranscription.textContent?.trim() || '';
    if (this.rawTranscription.classList.contains('placeholder-active') || rawText === '') {
        console.warn('No raw transcription content to copy.');
        return;
    }
    
    try {
        await navigator.clipboard.writeText(rawText);
        this.setButtonState(this.copyRawButton, 'success');
    } catch (err) {
        console.error('Failed to copy raw transcription: ', err);
        this.setButtonState(this.copyRawButton, 'error');
    }
  }

  private async copyMetadata(): Promise<void> {
      if (!this.currentNote || this.currentNote.duration === 0 && this.totalDurationMs === 0) {
          console.warn('No metadata to copy.');
          return;
      }
      const { timestamp, audioSize, modeId, cost } = this.currentNote;
      
      const metaString = [
          `Date & Time: ${new Date(timestamp).toLocaleString(undefined, { year: 'numeric', month: 'long', day: 'numeric', hour: 'numeric', minute: '2-digit', timeZone: this.getIanaTimezone(this.currentTimezone)})}`,
          `Recording Duration: ${this.formatDuration(this.totalDurationMs || this.currentNote.duration)}`,
          `Audio File Size: ${this.formatBytes(audioSize)}`,
          `Processing Mode: ${MODES[modeId].name}`,
          `Estimated Cost (USD): $${cost.toFixed(5)}`
      ].join('\n');

      try {
          await navigator.clipboard.writeText(metaString);
          this.setButtonState(this.copyMetaButton, 'success');
      } catch (err) {
          console.error('Failed to copy metadata: ', err);
          this.setButtonState(this.copyMetaButton, 'error');
      }
  }

  private downloadFullAudio(): void {
    if (this.sessionAudioChunks.length === 0) {
      console.warn('No audio chunks recorded to download.');
      this.setButtonState(this.downloadAudioButton, 'error');
      const message = 'Audio missing...';
      this.setGlobalStatus(message, false, true);
      setTimeout(() => {
        if (this.globalStatus.textContent?.includes(message)) {
            this.setGlobalStatus('Ready to record');
        }
      }, 3000);
      return;
    }
  
    const fullAudioBlob = new Blob(this.sessionAudioChunks, { type: this.sessionMimeType });
    const timestamp = new Date().toISOString().slice(0, 19).replace(/[-:T]/g, '');
    
    // Determine file extension from mime type
    let extension = 'webm';
    if (this.sessionMimeType.includes('mp4')) {
        extension = 'mp4';
    } else if (this.sessionMimeType.includes('ogg')) {
        extension = 'ogg';
    }

    const filename = `voicenote-audio-${timestamp}.${extension}`;
  
    const link = document.createElement('a');
    link.href = URL.createObjectURL(fullAudioBlob);
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(link.href);

    this.setButtonState(this.downloadAudioButton, 'success');
  }
  
  private async downloadPolishedNote(): Promise<void> {
    if (!this.currentNote || !this.currentNote.polishedNote.trim()) {
        console.warn('No polished note.');
        this.setButtonState(this.downloadNoteButton, 'error');
        const message = 'Empty Export...';
        this.setGlobalStatus(message, false, true);
        setTimeout(() => {
            if (this.globalStatus.textContent?.includes(message)) {
                this.setGlobalStatus('Ready to record');
            }
        }, 3000);
        return;
    }

    const markdownContent = this.currentNote.polishedNote;
    const blob = new Blob([markdownContent], { type: 'text/markdown;charset=utf-8' });
    
    const timestamp = new Date(this.currentNote.timestamp).toISOString().slice(0, 19).replace(/[-:T]/g, '');
    const filename = `voicenote-${timestamp}.md`;

    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(link.href);
    
    this.setButtonState(this.downloadNoteButton, 'success');
  }

  private formatDuration(ms: number): string {
    if (ms <= 0) return '00:00';
    const totalSeconds = Math.floor(ms / 1000);
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
  }

  private formatBytes(bytes: number, decimals = 2): string {
    if (bytes <= 0) return '--';
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
  }

  private updateNoteCost(): void {
    if (!this.currentNote) return;
    const { promptTokens, completionTokens } = this.currentNote;
    const promptCost = (promptTokens / 1000) * COST_PER_1K_PROMPT_TOKENS;
    const completionCost = (completionTokens / 1000) * COST_PER_1K_COMPLETION_TOKENS;
    this.currentNote.cost = promptCost + completionCost;
  }
  
  private resetMetadataDisplay(): void {
    this.metaDatetime.querySelector('span')!.textContent = '--';
    this.metaDuration.querySelector('span')!.textContent = '--';
    this.metaSize.querySelector('span')!.textContent = '--';
    this.metaMode.querySelector('span')!.textContent = '--';
    this.metaCost.querySelector('span')!.textContent = '$0.00000';
  }

  private updateMetadataDisplay(isLive: boolean = false): void {
    if (!this.currentNote) {
        this.resetMetadataDisplay();
        return;
    };
    const { timestamp, audioSize, modeId, cost } = this.currentNote;
    const dtSpan = this.metaDatetime.querySelector('span')!;
    dtSpan.textContent = new Date(timestamp).toLocaleString(undefined, {
        year: 'numeric', month: 'long', day: 'numeric',
        hour: 'numeric', minute: '2-digit',
        timeZone: this.getIanaTimezone(this.currentTimezone),
    });
    const durSpan = this.metaDuration.querySelector('span')!;
    const duration = this.isRecording ? this.totalDurationMs : (this.currentNote.duration || this.totalDurationMs);
    durSpan.textContent = isLive ? 'Recording...' : this.formatDuration(duration);
    const sizeSpan = this.metaSize.querySelector('span')!;
    sizeSpan.textContent = isLive ? '...' : this.formatBytes(audioSize);
    const modeSpan = this.metaMode.querySelector('span')!;
    modeSpan.textContent = MODES[modeId].name;
    const costSpan = this.metaCost.querySelector('span')!;
    costSpan.textContent = (cost > 0) ? `$${cost.toFixed(5)}` : '$0.00000';
    if(this.currentNote) this.currentNote.duration = this.totalDurationMs;
  }

  private resetToIdleState(): void {
    this.isRecording = false;
    this.isPaused = false;
    this.isProcessing = false;
    this.hideRecordingDialog();
    
    if (this.stream) {
        this.stream.getTracks().forEach(track => track.stop());
        this.stream = null;
    }
    this.mediaRecorder = null;
    this.updateMetadataDisplay();
  }

  private setLiveControls(enabled: boolean): void {
    this.stopButton.disabled = !enabled;
    this.pauseButton.disabled = !enabled;
    this.lapButton.disabled = !enabled;
  }

  private createNewNote(): void {
    if(this.isRecording) {
        this.stopFullRecordingSession();
    }

    this.currentNote = {
      id: `note_${Date.now()}`,
      rawTranscription: '',
      polishedNote: '',
      timestamp: Date.now(),
      duration: 0,
      audioSize: 0,
      modeId: this.currentModeId,
      promptTokens: 0,
      completionTokens: 0,
      cost: 0,
    };
    
    this.allRawLapText = '';
    this.totalDurationMs = 0;
    this.downloadAudioButton.disabled = true;

    const rawPlaceholder = this.rawTranscription.getAttribute('placeholder') || '';
    this.rawTranscription.textContent = rawPlaceholder;
    this.rawTranscription.classList.add('placeholder-active');
    const polishedPlaceholder = this.polishedNote.getAttribute('placeholder') || '';
    this.polishedNote.innerHTML = polishedPlaceholder;
    this.polishedNote.classList.add('placeholder-active');

    this.resetMetadataDisplay();
    this.setGlobalStatus('Ready to record');
    this.resetToIdleState();
    this.initMoreMenu();
  }
}

document.addEventListener('DOMContentLoaded', () => {
  new VoiceNotesApp();
  document.querySelectorAll<HTMLElement>('[contenteditable][placeholder]').forEach((el) => {
      const placeholder = el.getAttribute('placeholder')!;
      function updatePlaceholderState() {
        const currentText = (el.id === 'polishedNote' ? el.innerText : el.textContent)?.trim();
        if (currentText === '' || currentText === placeholder) {
          if (el.id === 'polishedNote' && currentText === '') el.innerHTML = placeholder;
          else if (currentText === '') el.textContent = placeholder;
          el.classList.add('placeholder-active');
        } else {
          el.classList.remove('placeholder-active');
        }
      }
      updatePlaceholderState();
      el.addEventListener('focus', function () {
        const currentText = (this.id === 'polishedNote' ? this.innerText : this.textContent)?.trim();
        if (currentText === placeholder) {
          if (this.id === 'polishedNote') this.innerHTML = '';
          else this.textContent = '';
          this.classList.remove('placeholder-active');
        }
      });
      el.addEventListener('blur', () => updatePlaceholderState());
    });
});

export {};