# TIPA opensource toolkit project
TIPA: Thermal Imaging-based Physiological and Affective computing

Lead contributor: Dr. Youngjun Cho, Assistant Professor, Department of Computer Science, University College London (UCL)



## Brief guideline

1. Download Anaconda (latest version) - Python 3.7 (recommended)

    https://www.anaconda.com/distribution/


2. Install basic libraries on the Conda console.

    conda install -c conda-forge opencv
    
    conda install scikit-learn
    
    pip install --upgrade numpy
    
    pip install --upgrade matplotlib


    * For your information
    
        print(python_version()) 
        
        3.7.3
        
        print(np.version.version)
        
        1.16.4
        
        print(cv2.__version__)
        
        3.4.2


3. Run "TIPA_basic_run.ipynb" on the Jupyter notebook 

    You can find a basic instruction on the notebook.
    


    
    
## Key Reference
[1] Youngjun Cho and Nadia Bianchi-Berthouze. 2019. Physiological and Affective Computing through Thermal Imaging: A Survey. arXiv:1908.10307 [cs], http://arxiv.org/abs/1908.10307

### Further Technical References
[2] Cho, Y., Julier, S.J., Marquardt, N. and Bianchi-Berthouze, N., 2017. Robust tracking of respiratory rate in high-dynamic range scenes using mobile thermal imaging. Biomedical optics express, 8(10), pp.4480-4503. https://doi.org/10.1364/BOE.8.004480

[3] Cho, Y., Julier, S.J. and Bianchi-Berthouze, N., 2019. Instant Stress: Detection of Perceived Mental Stress Through Smartphone Photoplethysmography and Thermal Imaging. JMIR mental health, 6(4), p.e10140. https://doi.org/10.2196/10140

[4] Cho, Y., Bianchi-Berthouze, N. and Julier, S.J., 2017. DeepBreath: Deep learning of breathing patterns for automatic stress recognition using low-cost thermal imaging in unconstrained settings. In 2017 Seventh International Conference on Affective Computing and Intelligent Interaction (ACII) (pp. 456-463). IEEE. https://doi.org/10.1109/ACII.2017.8273639

[5] Cho, Y., Bianchi-Berthouze, N., Marquardt, N. and Julier, S.J., 2018. Deep Thermal Imaging: Proximate Material Type Recognition in the Wild through Deep Learning of Spatial Surface Temperature Patterns. In Proceedings of the 2018 CHI Conference on Human Factors in Computing Systems, ACM. https://doi.org/10.1145/3173574.3173576
