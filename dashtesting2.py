from dash import Dash, dcc, html
import plotly.express as px

fig = px.line(x=[1, 2, 3, 4], y=[1, 4, 9, 16], title=r'$\alpha_{1c} = 352 \pm 11 \text{ km s}^{-1}$')
fig.update_layout(
    xaxis_title=r'$\sqrt{(n_\text{c}(t|{T_\text{early}}))}$',
    yaxis_title=r'$d, r \text{ (solar radius)}$'
)

app = Dash(__name__)

app.layout = html.Div([

    dcc.Markdown('''
    ## LaTeX in a Markdown component:

    This example uses the block delimiter:
    $$
    \\frac{1}{(\\sqrt{\\phi \\sqrt{5}}-\\phi) e^{\\frac25 \\pi}} =
    1+\\frac{e^{-2\\pi}} {1+\\frac{e^{-4\\pi}} {1+\\frac{e^{-6\\pi}}
    {1+\\frac{e^{-8\\pi}} {1+\\ldots} } } }
    $$

    This example uses the inline delimiter:
    $E^2=m^2c^4+p^2c^2$

    ## LaTeX in a Graph component:

    ''', mathjax=True),

    dcc.Graph(mathjax=True, figure=fig)]
)

if __name__ == '__main__':
    app.run(debug=True)