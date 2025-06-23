# unlabelled-speaker-recognition

My solution mainly consist of designs. Codes inside the python notebook is mainly for me to quick test my ideas as direction of the solution only.

#### Data Sample from URL https://www.openslr.org/12

#### Diagram is created at draw.io

To investigate potential approaches for detecting and recognizing
individual speakers from a dataset of over 200 unlabelled microphone
recordings. When a voice is present, the goal is to identify which of the 200
speakers it belongs to.

Background: You have been provided with a dataset containing over
recordings from various microphone speakers. These recordings are completely
unlabelled; we do not know the identity of the speaker in any given segment.
Your task is to explore and propose a feasible solution for automatically
recognizing/labelling which of the 200 speakers is speaking.

## Data Exploration and Analysis [10%]:

- Describe the potential characteristics and significant challenges of working with
  unlabelled audio data for speaker recognition.

1. data cleaning - visuallize waveforms to cut out overly quiet samples- how do make a script to do this ? find avg of amplitude ?
2. data cleaning - check distribution of durations.. making sure every label have the same number of samples (3s)
3. Observe spectograms.
4. Observe prosodic features . .

## Proposed Solution(s) and Justification [50%]:

1. Audio to spectograms (2-3 seconds) and perform CNNs to get the embedding of pronouncetiation.. exlcude the classification layer.
2. Audio to prosodic vectors and perform RNNs to get the embedding to capture talking style.. exclude the classfication layer.
3. align time windows and concat
4. perform clustering and classification

### Why this make sense ?

AST zeroes in on spectral cues—phonemes, enunciation, etc.
Your prosody CNN zeroes in on suprasegmental cues—intonation, energy, pitch dynamics.
Fusing them has been effective in prior speech models

## Implementation Strategy (Conceptual) [20%]:

Conceptual Diagram

- How you would attempt to evaluate the potential success of your
  approach without ground truth speaker labels. Consider metrics or
  qualitative assessments you could use to infer the quality of the speaker
  groupings.

## Challenges and Considerations [20%]:

### Challenges

1. clustering to 200 labels

### Considerations

1. Recording quality is not filtered
2. one-shot method: raw input to Pre-trained embedding model - it might work well but less flexibility of tuning. thats why not continuing on this
3. sampling the audio waveform segments to speed up process
