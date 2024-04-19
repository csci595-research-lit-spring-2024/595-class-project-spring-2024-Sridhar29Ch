import trans_1day
import trans_1hr
import trans_5min
import ARIMA_ANN_1Hr
import ARIMA_ANN_1day
import ARIMA_ANN_5min
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import easygui
import random

def main():
    model = sys.argv[1];
    if model == 'Hybrid':
        duration = sys.argv[2]
        if duration == '1D':
            out=ARIMA_ANN_1day.main(sys.argv[3])
            plt.figure
            rx = range(0,len(out[0]))
            plt.plot(rx,out[0][100])
            plt.title('RMS  PLOT FOR PREDICTED ERROR: '+ "  "+sys.argv[3])
            plt.show()
            outfinal=np.mean(out[1]*random.randrange(80,100)/100)
            print("Next day predicted value for "+sys.argv[3]+" is: "+str(outfinal))
            print("Error margin for "+sys.argv[3]+"is: "+str(np.mean(out[0])))
        elif duration == '5Min':
            out =ARIMA_ANN_5min.main(sys.argv[3])
            plt.figure
            rx = range(0,len(out[0]))
            plt.plot(rx,out[0][100])
            plt.title('RMS  PLOT FOR PREDICTED ERROR'+ "  "+sys.argv[3])
            plt.show()
            outfinal=np.mean(out[1]*random.randrange(80,100)/100)
            print("Next day predicted value for "+sys.argv[3]+" is: "+str(outfinal))
            print("Error margin for "+sys.argv[3]+"is: "+str(np.mean(out[0])))
        elif duration == '1H':
            out = ARIMA_ANN_1Hr.main(sys.argv[3])
            plt.figure
            rx = range(0,len(out[0]))
            plt.plot(rx,out[0][100])
            plt.title('RMS  PLOT FOR PREDICTED ERROR'+ "  "+sys.argv[3])
            plt.show()
            outfinal=np.mean(out[1]*random.randrange(80,100)/100)
            print("Next day predicted value for "+sys.argv[3]+" is: "+str(outfinal))
            print("Error margin for "+sys.argv[3]+"is: "+str(np.mean(out[0])))
        else:
            print("Please enter valid time duration amoung 1D, 5Min, 1H")
    else:
        duration = sys.argv[2]
        if duration == '1D':
            out=trans_1day.main(sys.argv[3])
            plt.figure
            rx = range(0,len(out[0]))
            plt.plot(rx,out[0][100])
            plt.title('RMS  PLOT FOR PREDICTED ERROR'+ "  "+sys.argv[3])
            plt.show()
            outfinal=np.mean(out[1]*random.randrange(80,100)/100)
            print(outfinal)
            easygui.msgbox("The next day prection value is "+str(outfinal))
            print("Next day predicted value for "+sys.argv[3]+" is: "+str(outfinal))
            print("Error margin for "+sys.argv[3]+"is: "+str(np.mean(out[0])))
        elif duration == '5Min':
            out=trans_5min.main(sys.argv[3])
            plt.figure
            rx = range(0,len(out[0]))
            plt.plot(rx,out[0][100])
            plt.title('RMS  PLOT FOR PREDICTED ERROR'+ "  "+sys.argv[3])
            plt.show()
            outfinal=np.mean(out[1]*random.randrange(80,100)/100)
            print("Next day predicted value for "+sys.argv[3]+" is: "+str(outfinal))
            print("Error margin for "+sys.argv[3]+"is: "+str(np.mean(out[0])))
        elif duration == '1H':
            out=trans_1hr.main(sys.argv[3])
            plt.figure
            rx = range(0,len(out[0]))
            plt.plot(rx,out[0][100])
            plt.title('RMS  PLOT FOR PREDICTED ERROR'+ "  "+sys.argv[3])
            plt.show()
            outfinal=np.mean(out[1]*random.randrange(80,100)/100)
            print("Next day predicted value for "+sys.argv[3]+" is: "+str(outfinal))
            print("Error margin for "+sys.argv[3]+"is: "+str(np.mean(out[0])))
        else:
            print("Please enter valid time duration amoung 1D, 5Min, 1H")

    

main()