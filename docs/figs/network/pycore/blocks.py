
from .tikzeng import *

#define new block
def block_2ConvPool( name, botton, top, s_filer=256, n_filer=64, offset="(1,0,0)", size=(32,32,3.5), opacity=0.5 ):
    return [
    to_ConvConvRelu( 
        name="ccr_{}".format( name ),
        s_filer=str(s_filer), 
        n_filer=(n_filer,n_filer), 
        offset=offset, 
        to="({}-east)".format( botton ), 
        width=(size[2],size[2]), 
        height=size[0], 
        depth=size[1],   
        ),    
    to_Pool(         
        name="{}".format( top ), 
        offset="(0,0,0)", 
        to="(ccr_{}-east)".format( name ),  
        width=1,         
        height=size[0] - int(size[0]/4), 
        depth=size[1] - int(size[0]/4), 
        opacity=opacity, ),
    to_connection( 
        "{}".format( botton ), 
        "ccr_{}".format( name )
        )
    ]


def block_Unconv( name, botton, top, s_filer=256, n_filer=64, offset="(1,0,0)", size=(32,32,3.5), opacity=0.5 ):
    return [
        to_UnPool(  name='unpool_{}'.format(name),    offset=offset,    to="({}-east)".format(botton),         width=1,              height=size[0],       depth=size[1], opacity=opacity ),
        to_ConvRes( name='ccr_res_{}'.format(name),   offset="(0,0,0)", to="(unpool_{}-east)".format(name),    s_filer=str(s_filer), n_filer=str(n_filer), width=size[2], height=size[0], depth=size[1], opacity=opacity ),       
        to_Conv(    name='ccr_{}'.format(name),       offset="(0,0,0)", to="(ccr_res_{}-east)".format(name),   s_filer=str(s_filer), n_filer=str(n_filer), width=size[2], height=size[0], depth=size[1] ),
        to_ConvRes( name='ccr_res_c_{}'.format(name), offset="(0,0,0)", to="(ccr_{}-east)".format(name),       s_filer=str(s_filer), n_filer=str(n_filer), width=size[2], height=size[0], depth=size[1], opacity=opacity ),       
        to_Conv(    name='{}'.format(top),            offset="(0,0,0)", to="(ccr_res_c_{}-east)".format(name), s_filer=str(s_filer), n_filer=str(n_filer), width=size[2], height=size[0], depth=size[1] ),
        to_connection( 
            "{}".format( botton ), 
            "unpool_{}".format( name ) 
            )
    ]




def block_Res( num, name, botton, top, s_filer=256, n_filer=64, intra_offset="(0,0,0)", size=(32,32,3.5)):
    lys = []
    layers = [ *[ '{}_{}'.format(name,i) for i in range(num-1) ], top]
    for i, name in enumerate(layers): 
        offset = "(2,0,0)" if i == 0 else intra_offset
        ly = [ to_Conv( 
            name='{}'.format(name),       
            offset=offset,
            to="({}-east)".format( botton ),   
            s_filer=str(s_filer), 
            n_filer=str(n_filer), 
            width=size[2],
            height=size[0],
            depth=size[1]
            ),
            to_connection( 
                "{}".format( botton  ), 
                "{}".format( name ) 
                )
            ]
        botton = name
        lys+=ly
    
    # lys += [
    #     to_skip( of=layers[1], to=layers[-2], pos=1.25),
    # ]
    return lys


def block_ConvNormRelu(name, botton, s_filer=256, n_filer=64, offset="(1,0,0)", size=(32, 32, 3.5), stride=1, opacity=0.6, xlabeloc="0.5"):
    return [
        to_Conv(
            name=f"{name}_conv",
            offset=offset,
            to="({}-east)".format(botton),
            s_filer=str(s_filer) if stride!=1 else "",
            n_filer=str(n_filer),
            width=size[2],
            height=size[0],
            depth=size[1],
            stride=stride,
            xlabeloc=xlabeloc
        ),
        to_connection(
            "{}".format(botton),
            "{}".format(f"{name}_conv")
        ),
        to_ConvRelu(
            name=f"{name}",
            offset="(0,0,0)",
            s_filer=str(s_filer) if stride <= 1 else "",
            n_filer="",
            to="({}-east)".format(f"{name}_conv"),
            width=0.5,
            height=round(size[0] // stride),
            depth=round(size[1] // stride),
            opacity=opacity
            )
    ]

# def input_img(img_path, name, height, depth, s_filer, n_filer, width=.01, caption=" ", offset=str(0, 0, 0)):
#     return [
#         to_ConvRelu(name="_".join(name, "rear"), s_filer=s_filer, n_filer=n_filer, height=height, depth=depth, width=width, caption=caption, offset=offset),
#         to_input(img_path, to=offset, name="_".join(name, "front"))
#     ]


def my_block_Res(num, botton, s_filer=256, n_filer=64, size=(32, 32, 3.5)):
    first = to_Conv(
        "res_blocks_0",
        offset="(2,0,0)",
        to="({}-east)".format(botton),
        s_filer=str(s_filer),
        n_filer=str(n_filer),
        width=size[2],
        height=size[0],
        depth=size[1]
    )
    first_connect = to_connection(botton, "res_blocks_0")

    mid = to_rect(3, 4, "res_blocks_0-east", xshift=9, label="$\\times 6$", caption="$\pmb{\cdots}$", name="hdots", is_boder=False)
    first_mid_connect = to_connection("res_blocks_0", "hdots", dst_style=".")

    last = to_Conv(
        f"res_blocks_{num-1}",
        offset="(1.5,0,0)",
        to="({}.east)".format("hdots"),
        s_filer=str(s_filer),
        n_filer=str(n_filer),
        width=size[2],
        height=size[0],
        depth=size[1]
    )
    mid_last_connect = to_connection("hdots", "res_blocks_5", src_style=".")

    return [first, first_connect, mid, first_mid_connect, last, mid_last_connect]
    # return [first, last]
