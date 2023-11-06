import sys
sys.path.append('../')
# sys.path.append('figs/network')
from pycore.tikzeng import *
from pycore.blocks  import *


mult = 1
scale = 18 * mult
base_spat = 32 * mult
base_width = base_spat / scale
im_sz = base_spat * 0.2
t_int_off = -2
phys_model_off = -18

def get_arch(type="petit"):
    
    if type not in ["cut", "petit"]:
        raise Exception("Not Implemented!")
    glob_offset = (-3, 0, 0)
    
    # input
    input = [
        to_ConvRelu(name='any', s_filer=256, n_filer=1, height=base_spat, depth=base_spat, width=.01, offset=str(glob_offset)),
        to_input("../pan_real.png", to=str(glob_offset), name="input", width=im_sz, height=im_sz),
    ]

    # pan int  # TODO: add more space between temperature map and image
    if type=="cut":
        pan_int_blocks = []
    else:
        pan_int_blocks = [
            to_Concat(name="concat_pan", offset="(7,0,0)", to="({}-east)".format("input"), radius=2.5, opacity=0.6),
            to_connection("input", "concat_pan"),
            to_int(name="t_pan", offset=f"(0,{t_int_off},0)", to="({}-south)".format("concat_pan"), radius=2.5, opacity=0.6, caption="Pan"),
            to_connection("t_pan", "concat_pan", src_side="north", dst_side="south")
        ]

    # encoder:
    if type == "cut":
        encoder = [
            *block_ConvNormRelu("cnr_b0", "input", s_filer=256, n_filer=64, offset="(3.5,0,0)", size=(base_spat, base_spat, base_width), stride=1),
            *block_ConvNormRelu("cnr_b1", "cnr_b0", s_filer=256, n_filer=64, offset="(3,0,0)", size=(base_spat, base_spat, base_width), stride=2),
            *block_ConvNormRelu("cnr_b2", "cnr_b1", s_filer=128, n_filer=128, offset="(2.5,0,0)", size=(base_spat/2, base_spat/2, base_width*2), stride=2),
            to_rect(9.5, 11, center="cnr_b1-anchor", xshift=-1.85, label="Encoder")
        ]
    else:
        encoder = [
            to_rect(4, 2, center="concat_pan-east", xshift=10, caption="\\textbf{Encoder}", label="", name="encoder"),
            to_connection("concat_pan", "encoder", dst_style=".")
        ]

    # bottleneck:
    if type == "cut":
        bottleneck = [
            *my_block_Res(6, "cnr_b2", s_filer=64, n_filer=256, size=(base_spat / 4, base_spat / 4, base_width * 4)),
            to_rect(10.5, 6, center="hdots.center", xshift=-0.25, label="Bottleneck", name="bottleneck")
        ]
    else:
        bottleneck = [
            to_rect(4, 2, center="encoder.east", xshift=10, label="", caption="\\textbf{Bottleneck}", name="bottleneck"),
            to_connection("encoder", "bottleneck", src_style=".", dst_style=".")
        ]

    # mono int:
    if type == "cut":
        mono_int_blocks = []
    else:
        mono_int_blocks = [
            to_Concat(name="concat_mono", offset="(3,0,0)", to="({}.east)".format("bottleneck"), radius=2.5, opacity=0.6),
            to_connection("bottleneck", "concat_mono", src_style="."),
            to_int(name="t_mono", offset=f"(0,{t_int_off},0)", to="({}-south)".format("concat_mono"), radius=2.5, opacity=0.6, caption="Mono"),
            to_connection("t_mono", "concat_mono", src_side="north", dst_side="south")
        ]

    # decoder
    if type == "cut":
        decoder = [
            *block_ConvNormRelu("cnr_b4", "res_blocks_5", s_filer="", n_filer=256, offset="(2,0,0)", size=(base_spat/4, base_spat/4, base_width*4), stride=0.5, xlabeloc="0.3"),
            *block_ConvNormRelu("cnr_b5", "cnr_b4", s_filer="", n_filer=128, offset="(2,0,0)", size=(base_spat/2, base_spat/2, base_width*2), stride=0.5, xlabeloc="-0.75"),
            *block_ConvNormRelu("cnr_b6", "cnr_b5", s_filer="", n_filer=64, offset="(3,0,0)", size=(base_spat, base_spat, base_width), stride=1),
            to_rect(10, 11, center="cnr_b5-anchor", xshift=0.25, label="Decoder")
        ]
    else:
        decoder = [
            to_rect(4, 2, center="concat_mono-east", xshift=19, caption="\\textbf{Decoder}", label="", name="decoder"),
            to_connection("concat_mono", "decoder", dst_style=".")
        ]

    if type == "cut":
        physical_model = []
    else:
        pan2T = r"$\sqrt[\leftroot{5} 4]{\frac{\pmb{I_\mathit{pan}} - p^{(0)}_{c_\mathit{pan}}(\pmb{T_\mathit{pan}})}{p^{(1)}_{c_\mathit{pan}}(\pmb{T_\mathit{pan}})}}$"
        T2Mono = r"$p^{(1)}_{c_\mathit{mono}}(\pmb{T_\mathit{mono}}) \pmb{\hat{T}_\mathit{obj}}^4 + p^{(0)}_{c_\mathit{mono}}(\pmb{T_\mathit{mono}})$"
        physical_model = [
            to_rect(6, 4, center="t_pan-south", yshift=phys_model_off, caption=pan2T, label=r"$\mathit{Pan} \rightarrow \hat{T}_\mathit{obj}$", name="pan2temp", style="solid"),
            """\path (input-east) -- (concat_pan-west) coordinate[pos=0.25] (between_in_concat);
            \draw [connection]  (between_in_concat)   --  node {\midarrow} (pan2temp.west-|between_in_concat) --  node {\midarrow} (pan2temp.west);""",
            to_connection("t_pan", "pan2temp", dst_style=".", src_side="south", dst_side="north"),
            to_rect(6, 4, center="t_mono-south", yshift=phys_model_off, caption=T2Mono, label=r"$\hat{T}_\mathit{obj} \rightarrow Mono$", name="temp2mono", style="solid"),
            to_connection("pan2temp", "temp2mono", src_style=".", dst_style=".", label="$\pmb{\hat{T}_\mathit{obj}}$"),
            to_connection("t_mono", "temp2mono", dst_style=".", src_side="south", dst_side="north"),
            to_Conv("affine", s_filer=256, n_filer=2, offset="(4,0,0)", to="(temp2mono.east)", width=1, height=base_spat, depth=base_spat),
            to_connection("temp2mono", "affine", src_style=".", label="$\pmb{\hat{I}_\mathit{mono}}$"),
            to_Sum(name="pm_sum", offset="(0,0,0)", to="(decoder.east-|affine-north)", radius=2.5, opacity=0.6),
            to_connection("affine", "pm_sum", src_side="north", dst_side="south"),
            to_connection("decoder", "pm_sum", src_style="."),
            to_rect(32, 10, center="temp2mono.center", xshift=-8.5, yshift=-0.75, label=r"$\pmb{\tilde{G}_{\mathit{phys}}}$", fontsz="\Huge", name="phys_model")
        ]

    # output
    out_src = "pm_sum" if type == "petit" else "cnr_b6"
    output = [
        to_ConvRelu(name='output', s_filer=256, n_filer=1, height=base_spat, depth=base_spat, width=.01, offset="(3.5, 0, 0)", to="({}-east)".format(out_src)),
        to_input("../filt_fake.png", to="({}-anchor)".format("output"), name="out1", width=im_sz, height=im_sz),
        to_connection(of=out_src, to="output")
    ]

    # legend
    if type=="cut":
        legend = [
            to_Conv("convS1_legend", s_filer="", n_filer="", offset="(-8,-12,0)", to="(bottleneck.north)", width=1, height=6, depth=6, stride=1, caption=r"{$7 \times 7 \; \text{Conv}$ $\text{Stride}=1$}"),
            to_Conv("convS2_legend", s_filer="", n_filer="", offset="(4,0,0)", to="(convS1_legend-east)", width=1, height=6, depth=6, stride=2, caption=r"{$3 \times 3 \; \text{Conv}$ $\text{Stride}=2$}"),
            to_Conv("res_block_legend", s_filer="", n_filer="", offset="(4,0,0)", to="(convS2_legend-east)", width=1, height=6, depth=6, stride=2, caption="Residual Block"),
            to_Conv("trans_conv_legend", s_filer="", n_filer="", offset="(4,0,0)", to="(res_block_legend-east)", width=1, height=6, depth=6, stride=0.5, caption=r"{$3 \times 3 \; \text{Trans.}$ $\text{Conv}$}"),
            to_ConvRelu("res_block_legend", s_filer="", n_filer="", offset="(4,0,0)", to="(trans_conv_legend-east)", width=1, height=6, depth=6, caption="InstanceNorm + ReLU"),
        ]
    else:
        legend = [
            to_Concat(name="concat", offset="(-10,-4,0)", to="({}.south)".format("phys_model"), radius=2.5, opacity=0.6, caption="Concatenate"),            
            to_int(name="t_int", offset="(4,0,0)", to="({}-east)".format("concat"), radius=2.5, opacity=0.6, caption=r"\\Intrinsic\\Temperature"),
            to_Sum(name="sum", offset="(4,0,0)", to="({}-east)".format("t_int"), radius=2.5, opacity=0.6, caption="Summation"),
            to_Conv("affine", s_filer="", n_filer="", offset="(4,0,0)", to="(sum-east)", width=1, height=6, depth=6, caption="Affine Transform"),
            to_rect(1.5, 1.5, center="affine-south", xshift=15, yshift=1, label="Calibrated Transform", name="calib", style="solid", fontsz="\huge")

        ]
    # connect componenets:
    arch = [
        *input,
        *pan_int_blocks,
        *encoder,
        *bottleneck,
        *mono_int_blocks,
        *decoder,
        *physical_model,
        *output,
        *legend
    ]


    return arch


def main():
    namefile = str(sys.argv[1])
    arch = get_arch(namefile)
    to_generate(arch, namefile + '.tex')

if __name__ == '__main__':
    main()
    
