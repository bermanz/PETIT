
from functools import partial
import os

def to_head( projectpath ):
    pathlayers = os.path.join( projectpath, 'layers/' ).replace('\\', '/')
    return r"""
\documentclass[border=8pt, multi, tikz]{standalone} 
\usepackage{import}
\subimport{"""+ pathlayers + r"""}{init}
\usetikzlibrary{positioning}
\usetikzlibrary{3d} %for including external image 
\usepackage{amsbsy}
\usepackage{amsmath}
"""

def to_cor():
    return r"""
\def\ConvColor{rgb:yellow,5;red,2.5;white,5}
\def\ConvReluColor{rgb:yellow,5;red,5;white,5}
\def\PoolColor{rgb:red,1;black,0.3}
\def\UnpoolColor{rgb:magenta,5;black,20}
\def\ResBlocksColor{rgb:blue,5;black,30}
\def\StrideOneColor{rgb:blue,5;red,2.5;white,5}
\def\StrideTwoColor{rgb:red,1;black,0.3}
\def\TransConvColor{rgb:blue,2;green,1;black,0.3}
\def\AffineColor{rgb:yellow,5;red,2.5;white,5}
\def\FcColor{rgb:blue,5;red,2.5;white,5}
\def\FcReluColor{rgb:blue,5;red,5;white,4}
\def\SoftmaxColor{rgb:magenta,5;black,7}   
\def\SumColor{rgb:blue,5;green,15}
\def\ConcatColor{rgb:blue,5;red,2.5;white,5}
"""

def to_begin():
    return r"""
\newcommand{\copymidarrow}{\tikz \draw[-Stealth,line width=0.8mm,draw={rgb:blue,4;red,1;green,1;black,3}] (-0.3,0) -- ++(0.3,0);}

\begin{document}
\begin{tikzpicture}
\tikzstyle{connection}=[ultra thick,every node/.style={sloped,allow upside down},draw=\edgecolor,opacity=0.7]
\tikzstyle{copyconnection}=[ultra thick,every node/.style={sloped,allow upside down},draw={rgb:blue,4;red,1;green,1;black,3},opacity=0.7]
"""

# layers definition

def to_input( pathfile, to='(-3,0,0)', width=8, height=8, name="temp" ):
    return r"""
\node[canvas is zy plane at x=0] (""" + name + """) at """+ to +""" {\includegraphics[width="""+ str(width)+"cm"+""",height="""+ str(height)+"cm"+"""]{"""+ pathfile +"""}};
\coordinate ("""+name+"""-east) at """ + to + """;
"""

# Conv


def to_Conv(name, s_filer=256, n_filer=64, offset="(0,0,0)", to="(0,0,0)", width=1, height=40, depth=40, caption=" ", stride=1, xlabeloc=0.5):
    # select color:
    if "res_block" in name:
        color = r"\ResBlocksColor"
    elif "affine" in name:
        color = r"\AffineColor"
    else:
        if stride > 1:
            color = r"\StrideTwoColor"
        elif stride == 1:
            color = r"\StrideOneColor"
        else:  # transposed convlution
            color = r"\TransConvColor"

    return r"""
\pic[shift={"""+ offset +"""}] at """+ to +""" 
    {Box={
        name=""" + name +""",
        caption="""+ caption +r""",
        xlabel={{"""+ str(n_filer) +""", }},
        zlabel="""+ str(s_filer) +""",
        fill = """ + color + """,
        height="""+ str(height) +""",
        width="""+ str(width) +""",
        depth="""+ str(depth) +""",
        xlabeloc="""+ str(xlabeloc) +""",
        }
    };
"""

# Conv,relu
# Bottleneck
def to_ConvRelu( name, s_filer=256, n_filer=64, offset="(0,0,0)", to="(0,0,0)", width=2, height=40, depth=40, caption=" ", opacity=0.6):
    return r"""
\pic[shift={ """+ offset +""" }] at """+ to +""" 
    {RightBandedBox={
        name="""+ name +""",
        caption="""+ caption +""",
        xlabel={{"""+ str(n_filer) +""", }},
        zlabel="""+ str(s_filer) +""",
        fill=\ConvColor,
        bandfill=\ConvReluColor,
        height="""+ str(height) +""",
        width="""+ str(width) +""",
        depth="""+ str(depth) +""",
        opacity = """ + str(opacity) + """,
        bandopacity = """ + str(opacity) + """
        }
    };
"""

# Conv,Conv,relu
# Bottleneck
def to_ConvConvRelu( name, s_filer=256, n_filer=64, offset="(0,0,0)", to="(0,0,0)", width=(2,2), height=40, depth=40, caption=" " ):
    return r"""
\pic[shift={ """+ offset +""" }] at """+ to +""" 
    {RightBandedBox={
        name="""+ name +""",
        caption="""+ caption +""",
        xlabel={{"""+ str(n_filer) +""", }},
        zlabel="""+ str(s_filer) +""",
        fill=\ConvColor,
        bandfill=\ConvReluColor,
        height="""+ str(height) +""",
        width={ """+ str(width[0]) +""" , """+ str(width[1]) +""" },
        depth="""+ str(depth) +"""
        }
    };
"""



# Pool
def to_Pool(name, offset="(0,0,0)", to="(0,0,0)", width=1, height=32, depth=32, opacity=0.5, caption=" "):
    return r"""
\pic[shift={ """+ offset +""" }] at """+ to +""" 
    {Box={
        name="""+name+""",
        caption="""+ caption +r""",
        fill=\PoolColor,
        opacity="""+ str(opacity) +""",
        height="""+ str(height) +""",
        width="""+ str(width) +""",
        depth="""+ str(depth) +"""
        }
    };
"""

# unpool4, 
def to_UnPool(name, offset="(0,0,0)", to="(0,0,0)", width=1, height=32, depth=32, opacity=0.5, caption=" "):
    return r"""
\pic[shift={ """+ offset +""" }] at """+ to +""" 
    {Box={
        name="""+ name +r""",
        caption="""+ caption +r""",
        fill=\UnpoolColor,
        opacity="""+ str(opacity) +""",
        height="""+ str(height) +""",
        width="""+ str(width) +""",
        depth="""+ str(depth) +"""
        }
    };
"""



def to_ConvRes( name, s_filer=256, n_filer=64, offset="(0,0,0)", to="(0,0,0)", width=6, height=40, depth=40, opacity=0.2, caption=" " ):
    return r"""
\pic[shift={ """+ offset +""" }] at """+ to +""" 
    {RightBandedBox={
        name="""+ name + """,
        caption="""+ caption + """,
        xlabel={{ """+ str(n_filer) + """, }},
        zlabel="""+ str(s_filer) +r""",
        fill={rgb:white,1;black,3},
        bandfill={rgb:white,1;black,2},
        opacity="""+ str(opacity) +""",
        height="""+ str(height) +""",
        width="""+ str(width) +""",
        depth="""+ str(depth) +"""
        }
    };
"""


# ConvSoftMax
def to_ConvSoftMax( name, s_filer=40, offset="(0,0,0)", to="(0,0,0)", width=1, height=40, depth=40, caption=" " ):
    return r"""
\pic[shift={"""+ offset +"""}] at """+ to +""" 
    {Box={
        name=""" + name +""",
        caption="""+ caption +""",
        zlabel="""+ str(s_filer) +""",
        fill=\SoftmaxColor,
        height="""+ str(height) +""",
        width="""+ str(width) +""",
        depth="""+ str(depth) +"""
        }
    };
"""

# SoftMax
def to_SoftMax( name, s_filer=10, offset="(0,0,0)", to="(0,0,0)", width=1.5, height=3, depth=25, opacity=0.8, caption=" " ):
    return r"""
\pic[shift={"""+ offset +"""}] at """+ to +""" 
    {Box={
        name=""" + name +""",
        caption="""+ caption +""",
        xlabel={{" ","dummy"}},
        zlabel="""+ str(s_filer) +""",
        fill=\SoftmaxColor,
        opacity="""+ str(opacity) +""",
        height="""+ str(height) +""",
        width="""+ str(width) +""",
        depth="""+ str(depth) +"""
        }
    };
"""

def to_ball(name, offset="(0,0,0)", to="(0,0,0)", radius=2.5, opacity=0.6, logo="$||$", caption="", color="\ConcatColor", scale=1):
    return r"""
\pic[shift={""" + offset + """}] at """ + to + """ 
    {Ball={
        name =""" + name + """,
        fill= """ + color + """,
        opacity = """ + str(opacity) + """,
        radius = """ + str(radius) + """,
        logo= """ + logo + """,
        caption= """ + caption + """,
        scale= """ + str(scale) + """
        }
    };
"""


to_PhyiscalModel = partial(to_ball, radius=20, logo="", color=r"\SoftmaxColor", caption="Physical Model", scale=0.25)
to_Concat = partial(to_ball, logo="$\pmb{||}$", color="\ConcatColor")
to_Sum = partial(to_ball, logo="$\pmb{+}$", color="\SumColor")
to_int = partial(to_ball, logo="$\pmb{T}$", color="\PoolColor")


def to_connection( of, to, src_side="east", dst_side="west", src_style="-", dst_style="-", label=None):
    if label is None:
        label_str = ""
    else:
        label_str = "node[above] {\huge " + label + "}"
    return r"""
\draw [connection]  ("""+of + src_style + src_side + """)    -- node {\midarrow} """ + label_str + """ ("""+to + dst_style + dst_side + """);
"""

def to_skip( of, to, pos=1.25):
    return r"""
\path ("""+ of +"""-southeast) -- ("""+ of +"""-northeast) coordinate[pos="""+ str(pos) +"""] ("""+ of +"""-top) ;
\path ("""+ to +"""-south)  -- ("""+ to +"""-north)  coordinate[pos="""+ str(pos) +"""] ("""+ to +"""-top) ;
\draw [copyconnection]  ("""+of+"""-northeast)  
-- node {\copymidarrow}("""+of+"""-top)
-- node {\copymidarrow}("""+to+"""-top)
-- node {\copymidarrow} ("""+to+"""-north);
"""


def to_pm_connect(model_name):
    if model_name == "petit":
        return r"""
            %% Physical model inputs:

            % pan image
            \path (input-east) -- (concat_pan-west) coordinate[pos=0.25] (between_in_concat) ;
            \draw [connection]  (between_in_concat)   --  node {\midarrow} (pm-west-|between_in_concat) --  node {\midarrow} (pm-west);

            % pan ambient
            \draw [connection]  (t_pan-south)   --  node {\midarrow} (pm-northwest-over-|t_pan-south) --  node {\midarrow} (pm-northwest-over) --  node {\midarrow} (pm-northwest);

            % mon ambient
            \draw [connection]  (t_mono-south)   --  node {\midarrow} (pm-northeast-over-|t_mono-south) --  node {\midarrow} (pm-northeast-over) --  node {\midarrow} (pm-northeast);

            %% Physical model output:
            \draw [connection]  (pm-east)   --  node {\midarrow} (pm-east-|pm_sum-south) --  node {\midarrow} (pm_sum-south);
            """
    else:
        return """"""


def to_rect(width, height, center, xshift=0, yshift=0, label="", caption="", style="dashed", name="r", is_boder=True, fontsz="\Huge"):
    draw = "draw,\n" if is_boder else ""
    return rf"""\node[rectangle,
        """ + draw + """""" + style + """,
        ultra thick,
        minimum width = """ + str(width) + """cm,
        minimum height = """ + str(height) + """cm,        
        label=below:""" + fontsz + " {" + label + """}](""" + name + """) at([xshift=""" + str(xshift) + """em, yshift=""" + str(yshift) + """em]""" + center + """) {""" + fontsz + """ """ + caption + """};
    """

def to_end():
    return r"""
\end{tikzpicture}
\end{document}
"""


def to_generate(arch, pathname="file.tex"):
    arch_wrapp = [to_head('..'), to_cor(), to_begin(), *arch, to_end()]
    with open(pathname, "w") as f: 
        for c in arch_wrapp:
            print(c)
            f.write( c )
     


