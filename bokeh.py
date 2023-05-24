from bokeh.plotting import figure , output_file , show

output_file('bokeh.html')
x1=[1,2,3]
x2=[4,8,3]

p=figure()
p.line(x1,x2,color='red')

show(p)