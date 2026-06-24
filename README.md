- create a env 
- pip install -r requirements.txt
- python shcnndata.py
- python vis.py --frame={frame_no.}

If Python 3.10 fails on the full requirements file, install the minimal
wavefront reconstruction dependencies instead:

- pip install -r requirements-base.txt

Base matrix reconstruction model:

- python -m BaseModal.train --file Sum_NewData_299_100.h5 --modes 100
- python -m BaseModal.predict --file Sum_NewData_299_100.h5 --frame={frame_no.}
- python -m BaseModal.visualize_prediction --file Sum_NewData_299_100.h5 --frame={frame_no.}
- python -m BaseModal.evaluate --file Sum_NewData_299_100.h5
