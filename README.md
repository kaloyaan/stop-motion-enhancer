# stop-motion-enhancer
 Input a stop motion sequence and get a video that's deflickered, stabilized, and doubled in frames. Project for CPSC678, Spring 2021 at Yale.

# How to run

1. Install dependencies (listed in requirements.txt)
2. To run the GUI, use the following command:
    
    `pythonw enhance_stopmotion.py`

In the case you're using a virtual environment such as conda on MacOS, there is a known issue with the wxPython library that might not let you launch the interface. A workaround is posted [here](https://stackoverflow.com/questions/48531006/wxpython-this-program-needs-access-to-the-screen).

3. If the GUI isn't launching, you can access the no-gui version using

    `python enhance_stopmotion_nogui.py`