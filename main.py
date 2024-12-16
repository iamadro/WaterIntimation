from nicegui import ui
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io
import base64
import plotly.graph_objects as go
import pandas as pd
import pickle as pkl
curr_cap = 1000
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

data = pkl.load(open('model.pkl', 'rb'))
model = data['model']
predicted_time_until_refill = model.predict(np.array(20).reshape(-1,1))[0]
refill_time = timedelta(hours=predicted_time_until_refill)
# print(f"Next refill time prediction: {refill_time.strftime('%I:%M %p')}")
real_time = datetime.now() + refill_time
if real_time.hour <= 12:
    pred_time = str(real_time.hour) + " : " + str(real_time.minute) + " AM"
else:
    pred_time = str(real_time.hour - 12) + " : " + str(real_time.minute) + " PM"

top_values = ["700 L", "600 L", f"{pred_time}"]
top_labels = ["Water used today", "Water refilled today", "Next refill time prediction"]

with ui.element('body').style('width: 100vw, height:100vh; margin: 0px'):
    # with ui.row().classes('w-full h-20').style('display: flex; align-items: center; justify-content: center; background: rgb(35, 148, 222)'):
    with ui.card().classes('w-full h-20').style('display: flex; align-items: center; justify-content: center; background: rgb(35, 148, 222); margin: 0px 0px 10px 0px'):
            ui.label(f'Water Intimation Using AI Model (Team-2)').style('color:white;font-size: 2.5vw; font-weight: bold; white-space: nowrap; padding: 10px')

    with ui.row().style('display: flex; gap:10px; flex-wrap: wrap;'):
        for i in range(3):
            with ui.card().tight().style('flex: 1'):
                with ui.row().style('height: 10vh;display: flex; align-items: center; flex-wrap: nowrap'):
                    ui.label(f'{top_values[i]}').style('font-size: 2.5vw; font-weight: bold; white-space: nowrap; padding: 10px')
                    ui.label(f'{top_labels[i]}').style('font-size: 1.5vw; white-space: nowrap;')

    with ui.row().style('height: 40vh; margin-top:10px; gap:1%; display:flex; justify-content: space-evenly'):
        with ui.card().style('flex: 2 1; height: 100%; float: left; padding: 0.5%; justify-content: center'):
            data = pd.read_csv('Water_Tank_Level_Prediction2.csv')
            times = data.iloc[:, 0].tolist()  # Time values as strings
            stored = data.iloc[:, 1].tolist()  # Water level as numeric

            fig = go.Figure(go.Scatter(x=times[::2], y=stored[::2]))
            fig.update_layout(title=dict(
                text="<b>Water Level Throughout the Day</b>", font=dict(size=18), x=0.5),
                margin=dict(l=0, r=0, t=36, b=0))
            ui.plotly(fig).classes('w-full h-full')  
          
        with ui.card().style('flex: 1 1; height: 100%; float: left; padding: 0.5%; object-fit: cover; display:flex; justify-content:center'):
            water_remained = 467
            ui.highchart({
                'title': {'text': 'Water remaining in tank (in L)'},
                'chart': {'type': 'solidgauge'},
                'yAxis': {
                    'min': 0,
                    'max': curr_cap,
                },
                'series': [
                    {'name': 'Water Remaining', 'data': [water_remained]},
                ],
            }, extras=['solid-gauge']).classes('w-full h-full')

        with ui.card().style('flex: 1 1; height: 100%; padding: 0.5%; float: left; align-items: center; justify-content: center'):
            # Define water usage by levels
            water_used_by_levels = [200, 150, 300, 100]  # Example data
            level_names = ['Ground Floor', 'First Floor', 'Second Floor', 'Third Floor']

            # Pie chart to represent water usage
            ui.highchart({
                'title': {'text': 'Water Usage by Levels'},
                'chart': {'type': 'pie'},
                'tooltip': {'pointFormat': '{series.name}: <b>{point.percentage:.1f}%</b>'},
                'plotOptions': {
                    'pie': {
                        'allowPointSelect': True,
                        'cursor': 'pointer',
                        'dataLabels': {
                            'enabled': True,
                            'format': '<b>{point.name}</b>: {point.y}L ({point.percentage:.1f}%)',
                        },
                    },
                },
                'series': [{
                    'name': 'Water Used',
                    'colorByPoint': True,
                    'data': [
                        {'name': level_names[i], 'y': water_used_by_levels[i]}
                        for i in range(len(level_names))
                    ],
                }],
            }).classes('w-full h-full')
    
    with ui.row().style('height: 40vh; margin-top:10px; gap:1%; display:flex; justify-content: space-evenly; flex-wrap: wrap'):
        with ui.card().style('flex: 2 1; height: 100%; float: left; padding: 0.5%; justify-content: center'):
            used = [1879, 1005, 1526, 1464, 1486, 1786, 1126, 1675, 1398, 1315, 1890, 995, 1877, 1196, 872]
            refilled = [1894, 1156, 1474, 1428, 1632, 1735, 1195, 1559, 1356, 1166, 1738, 1006, 1950, 1289, 822]
            chart = ui.highchart({
                'title': {'text': 'Last 15 days report'},
                'chart': {'type': 'column'},
                'xAxis': {'categories': ["Day "+str(i) for i in range(1,16)]},
                'series': [
                    {'name': 'Water Used (in L)', 'data': used[:15]},
                    {'name': 'Water Refilled (in L)', 'data': refilled[:15]},
                ],
            }).classes('w-full h-full')

        with ui.card().style('flex: 1 1; height: 100%; float: left; padding: 0.5%; object-fit: cover; display:flex; justify-content:center'):
            # Define water usage by levels
            water_used_by_levels = [100, 300, 400, 200, 50]  # Example data
            level_names = ['Kitchens', 'Washrooms', 'Gardening', 'Outdoors', 'Miscellaneous']

            # Pie chart to represent water usage
            ui.highchart({
                'title': {'text': 'Water Usage by Category'},
                'chart': {'type': 'pie'},
                'tooltip': {'pointFormat': '{series.name}: <b>{point.percentage:.1f}%</b>'},
                'plotOptions': {
                    'pie': {
                        'allowPointSelect': True,
                        'cursor': 'pointer',
                        'dataLabels': {
                            'enabled': True,
                            'format': '<b>{point.name}</b>: {point.y}L ({point.percentage:.1f}%)',
                        },
                    },
                },
                'series': [{
                    'name': 'Water Used',
                    'colorByPoint': True,
                    'data': [
                        {'name': level_names[i], 'y': water_used_by_levels[i]}
                        for i in range(len(level_names))
                    ],
                }],
            }).classes('w-full h-full')

        with ui.card().style('flex: 1 1; padding: 0.5%; height: 100%; float: left; align-items: center'):
            # ui.label("Group members:").style('font-size: 1.5vw; font-weight:bold')
            # with ui.grid(columns = 2).style("font-size: 100%; white-space: nowrap; height:100%;"):
            #     regs = ['22BAI10030', '22BAI10110', '22BAI10323', '22BAI10354', '22BAI10413']
            #     names = ['Aditya Routh', 'Jay Shayam Gowda', 'Aditya Raj Singh', 'Sawari Jamgaonkar', 'Ritam Goswami']
            #     for i in range(len(regs)):
            #         ui.label(regs[i])
            #         ui.label(names[i])

            exp_icons = ['calendar_month','leaderboard', 'data_usage']
            exp_labels = ['Day-wise usage', 'Floor-wise insights', 'Usage-wise insights']
            exp_desc = ['A lot of water got used in day 13.',
            'Second floor has used the most amount of water',
            'The category using the most amount of water is: Gardening']

            ui.label("Insights:").style('font-size:18px; font-weight: bold')
            with ui.scroll_area().classes('w-full'):
                for i in range(len(exp_icons)):
                    with ui.expansion(f'{exp_labels[i]}', icon=f'{exp_icons[i]}').classes('w-full'):
                        with ui.row().classes('w-full'):
                            with ui.scroll_area().classes('w-full h-20'):
                                ui.label(f'{exp_desc[i]}')

ui.run()
