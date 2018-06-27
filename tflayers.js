const tf = require('@tensorflow/tfjs')
require('@tensorflow/tfjs-node')

const model = tf.sequential()
model.add(tf.layers.dense({
    units:8,
    inputShape:[2],
    activation:'relu'
}))

model.add(tf.layers.dense({
    inputShape:[8],
    units:8,
    activation:'relu'
}))

model.add(tf.layers.dense({
    units:1,
    inputShape:[8]
}))

model.compile({
    optimizer: tf.train.adam(0.01),
    loss:tf.losses.meanSquaredError

})

const xs = tf.tensor2d([[0,0],[0,1],[1,0],[1,1]])
const ys = tf.tensor2d([[0],[1],[1],[0]])

model.fit(xs,ys,{
    shuffle:true,
    epochs:100
}).then((hist)=>{
    console.log(hist.history.loss[hist.history.loss.length - 1]);
    model.predict(xs).print()
})
