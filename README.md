This dissertation encompasses all experimental studies conducted, comprising ten distinct programs. 
All programs were implemented using TensorFlow, with substantial utilization of official TensorFlow tutorial code. 
For instance, the waveform-to-spectrogram conversion program was directly adapted from the tutorials. 
We gratefully acknowledge these resources.

We will now present the execution sequence of these programs and their primary functions. 
For implementation details, please refer to the respective README files accompanying each program. 

The initial program, wave2spectrogram.py, implements Short-Time Fourier Transform (STFT) to convert raw waveforms into spectrograms. 
Beyond this core functionality, it provides comparative visualization of both time-domain waveforms and their corresponding frequency-domain spectrogram representations for distinct voice commands. 

The second program, T_MNIST.2.py, implements the convolutional kernels (including Klein Features) proposed by Love et al. (2023) for natural images, adapted here for MNIST and CIFAR-10 datasets. 
These datasets are loaded using TensorFlow's native dataset utilities. 
Notably, the implementation employs pre-computed kernel values for all features except Klein One Layer, which was excluded due to computational complexity constraints. 

The third program, Example of Speech Box.py, mainly utilizes Textgrid files to segment original sentences and paragraphs into phoneme fragments.

The fourth program, Example1.3 of Speech Box.py, is used to obtain the weight vectors of a convolutional neural network, preparing for subsequent feature extraction. 

The fifth program, Data Process0.3.py, performs density filtering on the obtained weight vectors, displays the graphical results after principal component analysis (PCA), and the outcomes of persistent homology. 
Due to limitations of the computational equipment, only a small portion of second-order homology was calculated. 

The sixth program, Data Process 0.4, obtains each vector from the principal component analysis (PCA) along with their corresponding principal component proportions. 
It also attempts to summarize the most probable convolution kernel distribution.

The seventh experiment, Example 2 of Speech Box.py, uses the features extracted by the aforementioned programs to summarize several convolution kernel models and compares them with traditional neural networks. 
(However, the results are even worse than those of traditional neural networks.)

The eighth experiment, Example2.1 of Speech Box, made an initial attempt at orthogonal feature convolution layers (OF) and found that they already slightly outperformed Kernel Feature (KF). 

