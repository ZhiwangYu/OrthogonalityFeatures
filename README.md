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

The seventh program, Example 2 of Speech Box.py, uses the features extracted by the aforementioned programs to summarize several convolution kernel models and compares them with traditional neural networks. 
(However, the results are even worse than those of traditional neural networks.)

The eighth program, Example2.1 of Speech Box, made an initial attempt at orthogonal feature convolution layers (OF), which already showed a slight improvement over Kernel Feature (KF). 
However, this experiment was discarded, and the results presented in the paper actually correspond to the ninth program (the main program). 
Notably, the program did not apply normalization, yet by using this program, better results can be achieved. 

The ninth program, Example 2.6 of Speech Box, serves as the primary experiment of this paper. 
Initially, the OF's (Orthogonal Feature) initial convolution kernels were appropriately adjusted. 
Subsequently, experiments were conducted under various conditions, including multiple datasets, Gaussian white noise, and whether phonemes were filtered (considering data balance). Notably, this experiment utilized audio normalization, which was not applied in the previous experiments. 

The final program, T_MNIST_4.1.py, integrates Orthogonal Features (OF) into image recognition, serving as an extension of T_MNIST_2.py. 

Finally, the author of this paper is a beginner in deep learning and has limited proficiency in coding. 
Most of the code required assistance with design. 
For instance, regarding the number of layers in the convolutional neural network (CNN), three layers were initially used to achieve higher accuracy. 
Later, it was modified to two layers, focusing on the algorithmic advantages of the new convolutional kernels. 
Additionally, the audio normalization step was introduced only after proposing the addition of noise. 
Nevertheless, the code remains highly versatile for the main results of this paper. 
Furthermore, in random selections, some experiments did not use seeds, primarily because the experimental results showed good stability (though under a signal-to-noise ratio (SNR) of 0dB, the performance of orthogonal features (OF) was less stable and might even underperform compared to traditional neural networks). 
Additionally, many programs in this paper show clear signs of modification, as most of them underwent various experimental attempts. 

Thank you for visiting the homepage of this project. If you have any suggestions or feedback, we warmly welcome your guidance and exchange of ideas. 
