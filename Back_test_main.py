from Decisions import Position_generotor
from back_testing_center import Back_test


class Assessment():
    def __init__(self) -> None:
        pass 
    
    def assess(self):
        position_gen = Position_generotor()
        backtest_centre = Back_test("^N225")
        #things needed 
        #1, prices returns based on taht 
        #2, positions
        data = backtest_centre.dataframe
        returns = backtest_centre.make_returns()
        positions = position_gen.make_positions(data)
        generated_data = backtest_centre.create_data_frame_for_calc_result(returns,positions)
        backtest_centre.make_result(generated_data)

if __name__ == "__main__":
    Assessment().assess()