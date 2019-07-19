package main

import (
	"os"

	tg "github.com/galeone/tfgo"
	"github.com/galeone/tfgo/image"
	"github.com/galeone/tfgo/image/filter"
	"github.com/galeone/tfgo/image/padding"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

func main() {
	root := tg.NewRoot()
	grayImg := image.Read(root, "airplane.png", 1)
	grayImg = grayImg.Scale(0, 255)

	// Edge detection using sobel filter: convolution
	Gx := grayImg.Clone().Convolve(filter.SobelX(root), image.Stride{X: 1, Y: 1}, padding.SAME)
	Gy := grayImg.Clone().Convolve(filter.SobelY(root), image.Stride{X: 1, Y: 1}, padding.SAME)
	convoluteEdges := image.NewImage(root.SubScope("edge"), Gx.Square().Add(Gy.Square().Value()).Sqrt().Value()).EncodeJPEG()

	Gx = grayImg.Clone().Correlate(filter.SobelX(root), image.Stride{X: 1, Y: 1}, padding.SAME)
	Gy = grayImg.Clone().Correlate(filter.SobelY(root), image.Stride{X: 1, Y: 1}, padding.SAME)
	correlateEdges := image.NewImage(root.SubScope("edge"), Gx.Square().Add(Gy.Square().Value()).Sqrt().Value()).EncodeJPEG()

	results := tg.Exec(root, []tf.Output{convoluteEdges, correlateEdges}, nil, &tf.SessionOptions{})

	file, _ := os.Create("convolved.png")
	file.WriteString(results[0].Value().(string))
	file.Close()

	file, _ = os.Create("correlated.png")
	file.WriteString(results[1].Value().(string))
	file.Close()
}
