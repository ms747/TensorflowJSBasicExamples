const tf = require('@tensorflow/tfjs')
require('@tensorflow/tfjs-node')
// require("@tensorflow/tfjs-node-gpu"1)

const m = tf.variable(tf.scalar(Math.random()))
const b = tf.variable(tf.scalar(Math.random()))
const optimizer = tf.train.sgd(0.01)


function model(x) {
    // y = mx+b
    return m.mul(x).add(b)
}

function loss(pred,ys){
    return pred.sub(ys).square().mean()
}

function train(xs,ys,its=750){
    for(let i = 0 ; i < its ; i++){
        optimizer.minimize(()=>{
            const predys = model(xs)
            return loss(predys,ys)
        })   
    }
}

const xs = tf.tensor2d([[0,0],[0,1],[1,0],[1,1]])
const ys = tf.tensor2d([[0],[1],[1],[0]])


model(xs).print()
train(xs,ys)
model(xs).print()

console.log(tf.memory().numTensors)
