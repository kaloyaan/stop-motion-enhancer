# Stop Motion Enhancer

Stop Motion Enhancer is a tool for improving amateur stop motion animation through deflickering, stabilization and frame interpolation.

Its simple graphical user interface makes it particularly suitable for beginner filmmakers that are experimenting with stop motion and want better visual results. The results are noticeably smoother and have greater temporal coherency due to the adjusted colors, minimized camera shakes and doubling of framerate.

Project for CPSC678, Spring 2021 at Yale.
 
# Paper

Full implementation, challenges & future work are described in [the paper for the project](https://drive.google.com/file/d/1j5p9KDSPUUO5oPPTm8ZWSkbNLIJMqAnH/view?usp=sharing).

# Thanks

This code is primarily a proof of concept for the novel pipeline and the one-click GUI which makes these algorithms easy to use for general users. The stop motion enhancement here is made possible by adapting the following existing code. Thank you to the authors:

 - Deflickering - by [Gonçalo Martins](https://github.com/gondsm/timelapse_deflickerer)
 - Stabilization - by [Abhishek Singh Thakur](https://learnopencv.com/video-stabilization-using-point-feature-matching-in-opencv/)
 - Frame interpolation - using [RIFE-HDv2 by Huang et al.](https://arxiv.org/abs/2011.06294)

# How to run

1. Install dependencies (listed in requirements.txt)
2. To run the GUI, use the following command:
    
    `pythonw enhance_stopmotion.py`

In the case you're using a virtual environment such as conda on MacOS, there is a known issue with the wxPython library that might not let you launch the interface. A workaround is posted [here](https://stackoverflow.com/questions/48531006/wxpython-this-program-needs-access-to-the-screen).

3. If the GUI isn't launching, you can access the no-gui version using

    `python enhance_stopmotion_nogui.py`
