import streamlit as st
#import plotly.express as px
import pandas as pd
#import matplotlib.pyplot as plt
from bokeh.plotting import figure
from bokeh.io import curdoc
from bokeh.themes import built_in_themes
from scipy.interpolate import interp1d, splev, splrep
from scipy.integrate import trapz, quad
import numpy as np


st.set_page_config(page_title='Curve Fitting Toolbox')
st.title('Curve Fitting Toolbox')
st.write('By *Barak Kashi*')
st.write('##')

# --- DATA ---
with st.container():
	df = pd.read_csv('/home/bk/Desktop/data_sine', header=None)
	xdata, ydata = df.loc[:,0].tolist(), df.loc[:,1].tolist()
	data_col1, data_col2 = st.columns(2)
	data_file = data_col1.file_uploader(label="", label_visibility='collapsed')
	data_col1.button('Paste from clipboard', disabled=True)
	df.columns = ['x', 'y']
	data_col2.dataframe(df, width=360, height=200)
	try:
		df = pd.read_csv(data_file, header=None)
		df.sort_values(df.columns[0], inplace=True)
		df.style.set_properties(**{'text-align': 'left'})
		xdata, ydata = df.loc[:,0].tolist(), df.loc[:,1].tolist()
	except:
		pass
		#st.write('Failed to read CSV data!')
st.write('##')


# --- MODELS ---
with st.container():
	model = st.selectbox(label='**Model**', options=['Fourier series', 'tanh', 'User defined'])
	if model == 'Fourier series':
		col1, col2, col3, col4, col5 = st.columns(5)
		N = col1.number_input('Number of terms', step=1, min_value=1, max_value=100)
		interp_type = col2.selectbox(label='Interpolation type', options=['Linear', 'spline'])
	if model == 'tanh':
		pass
	if model == 'User defined':
		st.text_input('y(x) =', disabled=True)
st.write('##')


fig = figure()
curdoc().theme = 'dark_minimal'
curdoc().add_root(fig)
fig.circle(xdata, ydata,radius=0.16, legend_label="Data")


# --- PROCESS ---
if model == 'Fourier series':
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
	xspace = np.linspace(min(xdata), max(xdata), len(xdata)*10)
	#A = [round(A*10**digits)/10**digits for A in A]
	#B = [round(B*10**digits)/10**digits for B in B] 					 
	fig.line(xspace, [ A[0]/2 + sum(A[n]*np.cos(m*n*x) + B[n]*np.sin(m*n*x) for n in range(1, N+1)) for x in xspace ], line_width=3.5, line_color='green', legend_label="Model")
	output = 'A: '
	for a in A:
		output += str(a)+', '
	output = output[0:-2]
	output += '\nB: '
	for b in B:
		output += str(a)+', '
	output = output[0:-2]
		
		
		
st.bokeh_chart(fig, use_container_width=True)
col1, col2, col3, col4 = st.columns(4)
logx = col1.checkbox('LogX')
logy = col2.checkbox('LogY')
exptap_back = col3.number_input('Ext. back', min_value=0)
exptap_fore = col4.number_input('Ext. fore', min_value=0)



#fig,ax = plt.subplots()
#ax.plot(xdata, ydata, 'o')
#ax.plot([1,20], [30,50], '-')
#st.pyplot(fig)
#fig = px.scatter(df['x'], df['y'])
#st.write(fig)	

#fig.line([1,20], [30,50], line_width=2)


	
