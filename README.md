# unlabelled-speaker-recognition

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

1. Randomly listening of some recordings to identify issues and possible preprocessing steps
2. visuallize waveforms to cut out overly quiet samples- how do make a script to do this ? find avg of amplitude ?
3. Observe spectograms. to lead to cnn
4. check distribution of durations.. sample it to the same length as we want to create- embeddings

## Proposed Solution(s) and Justification [50%]:

1. Audio to spectograms (2-3 seconds) and perform CNNs to get the embedding of pronouncetiation.. exlcude the classification layer.
2. Audio to prosodic vectors and perform RNNs to get the embedding to capture talking style.. exclude the classfication layer.
3. align time windows and concat
4. perform clustering and classification

## Implementation Strategy (Conceptual) [20%]:

Here is the complete conceptual diagram for both of the speech recognition models.

## Challenges and Considerations [20%]:

### Challenges

1. concatenated embeddings might be highly overlapped. we change it to whole time.. spatial clustered might not be that obvious.

### Considerations

1. Recording quality is not filtered
2. raw input to Pre-trained embedding model - it might work well but less flexibility of tuning. thats why not continuing on this
