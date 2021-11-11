import sys
from streamlit import cli as stcli

if __name__ == '__main__':
    #sys.argv = ["streamlit", "run", "sampling.py"]
    sys.argv = ["streamlit", "run", "mnst_app.py"]
    sys.exit(stcli.main())