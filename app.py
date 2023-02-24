import streamlit as st
import pandas as pd
import numpy as np
from bokeh.plotting import figure
from bokeh.io import curdoc
from bokeh.themes import built_in_themes
from scipy.interpolate import interp1d, splev, splrep
from scipy.integrate import trapz, quad
from tkinter import Tk



# window & title setup
st.set_page_config(page_title='Fourier Curve Fitting')
st.title('Fourier Curve Fitting')
st.write('##')


@st.cache_data
def get_data_from_csv_file(file):
	return pd.read_csv(data_file, header=None)


# --- Get Data ---
data_col1, data_col2 = st.columns(2)
data_file = data_col1.file_uploader(label="", label_visibility='collapsed')
paste = data_col1.button('Paste from clipboard', disabled=False)
if paste:
	data_string = Tk().clipboard_get()
	df = pd.DataFrame([x.split(',') for x in data_string.split('\n')], columns=['x', 'y'])
elif data_file != None:
	df = get_data_from_csv_file(data_file)
	df.style.set_properties(**{'text-align': 'left'})
else:
	df = pd.read_csv('/home/bk/Desktop/data_sine', header=None)
	xdata, ydata = df.loc[:,0].tolist(), df.loc[:,1].tolist()
#st.write(df)
df.sort_values(df.columns[0], inplace=True)
df.columns = ['x', 'y']
xdata, ydata = df['x'].tolist(), df['y'].tolist()
xdata = [float(x) for x in xdata]
ydata = [float(y) for y in ydata]
data_col2.dataframe(df, width=360, height=200)
st.write('##')



# --- Fig. Setup ---
fig = figure(y_axis_type="log")
curdoc().theme = 'dark_minimal'
curdoc().add_root(fig)


# --- Plot Data ---
fig.circle(xdata, ydata, legend_label="Data")  #,radius=0.16


# --- Model Options ---
col1, col2, col3, col4 = st.columns(4)
N = col1.number_input('Number of terms', step=1, min_value=1, max_value=100)
interp_type = col2.selectbox(label='Interpolation type', options=['Linear', 'spline'])
extrap_back = col3.number_input('Extrapolate backward', min_value=0.0)
extrap_for = col4.number_input('Extrapolate forward', min_value=0.0)


P = xdata[-1] - xdata[0]
m = 2*np.pi/P
if interp_type == 'Linear':
	f = interp1d(xdata, ydata)
	fig.line(xdata, f(xdata), line_dash='dotted', line_width=1, line_color='magenta', legend_label="Interp.")
	A, B = [ 2/P*quad(f, xdata[0], xdata[-1])[0] ], [0]
	A_integrand = lambda x: f(x)*np.cos(m*n*x)
	B_integrand = lambda x: f(x)*np.sin(m*n*x)
else:  # spline
	smoothing_level = (len(xdata) - np.sqrt(len(xdata)))/2
	f = splrep(xdata, ydata, s=smoothing_level)
	interp_xspace = np.linspace(min(xdata), max(xdata), len(xdata)*10)	
	fig.line(interp_xspace, splev(interp_xspace, f), line_dash='dotted', line_width=1, line_color='magenta', legend_label="Interp.")
	spline = lambda x: splev(x, f)
	A, B = [ 2/P*quad(spline, xdata[0], xdata[-1])[0] ], [0]
	A_integrand = lambda x: splev(x, f)*np.cos(m*n*x)
	B_integrand = lambda x: splev(x, f)*np.sin(m*n*x)
for n in range(1, N+1):
	A.append( 2/P*quad(A_integrand, xdata[0], xdata[-1])[0] )
	B.append( 2/P*quad(B_integrand, xdata[0], xdata[-1])[0] )
xspace = np.linspace(min(xdata)-extrap_back, max(xdata)+extrap_for, len(xdata)*10)
fig.line(xspace, [ A[0]/2 + sum(A[n]*np.cos(m*n*x) + B[n]*np.sin(m*n*x) for n in range(1, N+1)) for x in xspace ], line_width=3.5, line_color='green', legend_label="Model")
st.bokeh_chart(fig, use_container_width=True)


# --- Write Coeffs. ---
digits = 3
A_out = [round(A*10**digits)/10**digits for A in A]
B_out = [round(B*10**digits)/10**digits for B in B] 					 
st.write(f'A = {A_out}')
st.write(f'B = {B_out}')

