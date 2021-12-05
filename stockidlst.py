import pandas as pd


class Stock_symbol_data_centre():
    def __init__(self) -> None:
        pass 

    def Get_data(self,run_mode=0):
        if run_mode ==0:
            sidlist = ["MMM", "AXP", "CAT","CVX","CSCO","KO","DOW","XOM","GS","HD","IBM","INTC","JNJ","JPM","MCD","MRK","MSFT","PFE","NKE","PG","TRV",
                    "UNH","VZ","WMT","WBA","DIS","^N225", "AAPL", "^GSPC", "TM", "GOOGL", "BA", "AMZN", "PYPL", "TWTR", "V"
                    ]
        elif run_mode == 1: 
            sp500_data_wiki =pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
            Data_frame = sp500_data_wiki[0]
            sidlist = Data_frame['Symbol'].values.tolist()
            sidlist.append("^N225")
            print(sidlist)
        return sidlist 


    def main(self):
        self.Get_data(1)

if __name__ == "__main__":
    Stock_symbol_data_centre().main()

#sidlist = ["VZ","WMT","WBA","DIS","^N225", "AAPL", "^GSPC", "TM", "GOOGL", "BA", "AMZN", "PYPL", "TWTR", "V"
          #]
#sidlist = ["^N225"]