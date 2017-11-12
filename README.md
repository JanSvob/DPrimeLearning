This repository contains Python framework for palmprint recognition using Convolutional Neural Networks, the main functionality is:

* learning filters from palmprint images
* learning embedding of the palmprint image into low dimensional feature vector 
* learning binary classifier comparing couples of images
* dataset processing and result evaluation

## Dependencies ##
[Lasagne](http://lasagne.readthedocs.org/en/latest/) - CNN implementation (layers, objectives, ...)

[Theano](http://deeplearning.net/software/theano/) - CNN implementation (symbolic expressions, ...)
 
[Seaborn](http://stanford.edu/~mwaskom/software/seaborn/) - visualization

[SciPy](http://www.scipy.org/) - image processing and visualization

[Bob](http://idiap.github.io/bob/) - biometrics (evaluation)

## Sample models ##
IITD model - trianed on the 1st half of the IITD dataset - leaves 2nd half for testing
Casia model - trained on the 1st half of the Casia dataset - leaves the 2nd half for testing
The two pre-trained models can be downloaded from here:
https://www.dropbox.com/s/gzv2heo9n43llfg/models.zip?dl=0

## Used datasets ##
Casia touchless palmprint database (2D) ([Info](http://biometrics.idealtest.org/dbDetailForUser.do?id=5))

IITD touchless palmprint database (2D) ([Info](http://www4.comp.polyu.edu.hk/~csajaykr/IITD/Database_Palm.htm))

## Related work ##
[[1]](http://www.jis.eurasipjournals.com/content/pdf/s13635-015-0022-z.pdf) R. Raghavendra and Ch. Busch - Texture based features for robust palmprint
recognition: a comparative study

[[2]](http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1467279) Z. Sun, T. Tan, Y. Wang and Stan Z. Li - Ordinal Palmprint Represention for Personal Identification

[[3]](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7335640) Q. Zheng, A. Kumar, and G. Pan - Suspecting Less and Doing Better: New Insights on Palmprint Identification for Faster and More Accurate Matching

[[3]](http://www.ntu.edu.sg/home/adamskong/publication/ICPR_2004.pdf) A. W. Kong and D. Zhang - Competitive Coding Scheme for Palmprint Verification

[[4]](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=5620980) V. Kanhangad, A. Kumar and D. Zhang - Contactless and Pose Invariant Biometric
Identification Using Hand Surface

[[5]](http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6105286) A. Morales, M. A. Ferrer and A. Kumar - Towards contactless palmprint authentication

[[6]](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=5670758) H. Imtiaz, S. A. Fattah - A DCT-based Feature Extraction Algorithm for Palm-print Recognition

[[7]](http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4756122) A. Kumar - Incorporating Cohort Information for Reliable Palmprint Authentication

[[8]](http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1227981) D. Zhang, A. W. Kong, J. You and M. Wong - Online Palmprint Identification

[[9]](http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1410434) A. Kumar and H. C. Shen  - Palmprint Identification Using PalmCodes

[[10]](http://www.ntu.edu.sg/home/adamskong/publication/ICBA_2004.pdf) A. W. Kong and D. Zhang - Feature-level Fusion for Effective Palmprint Authentication
