async function getData() {

    const carsDataResponse = await fetch('https://storage.googleapis.com/tfjs-tutorials/carsData.json');

    const carsData = await carsDataResponse.json();

    const cleaned = carsData.map(car => ({
        mpg: car.Miles_per_Gallon,
        horsepower: car.Horsepower
    }))
        .filter(car => (car.mpg != null && car.horsepower != null));

    return cleaned;

}

function createModel() {

    const model = tf.sequential();

    model.add(tf.layers.dense({ inputShape: [1], units: 1, useBias: true }));
    model.add(tf.layers.dense({ units: 50, activation: 'sigmoid'}));
    model.add(tf.layers.dense({ units: 50, activation: 'sigmoid'}));
    model.add(tf.layers.dense({ units: 50, activation: 'sigmoid'}));    
    
    model.add(tf.layers.dense({ units: 1, useBias: true }));

    return model;
}

function convertToTensor(data) {
    return tf.tidy(() => {
        tf.util.shuffle(data);

        const inputs = data.map(d => d.horsepower);
        const labels = data.map(d => d.mpg);

        const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
        const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

        const inputMax = inputTensor.max();
        const inputMin = inputTensor.min();
        const labelMax = labelTensor.max();
        const labelMin = labelTensor.min();


        const normalizedInputs =
            inputTensor.sub(inputMin)
                .div(inputMax.sub(inputMin));

        const normalizedLabels =
            labelTensor.sub(labelMin)
                .div(labelMax.sub(labelMin));


        return {
            inputs: normalizedInputs,
            labels: normalizedLabels,
            inputMax,
            inputMin,
            labelMax,
            labelMin,
        };

    });
}

async function trainModel(model, inputs, labels) {
    model.compile({
        optimizer: tf.train.adam(),
        loss: tf.losses.meanSquaredError,
        metrics: ["mse"]
    });

    const batchSize = 32;
    const epochs = 200;

    return await model.fit(inputs, labels, {
        batchSize,
        epochs,
        shuffle: true,
        callbacks: tfvis.show.fitCallbacks(
            { name: "Performance do treinamento" },
            ["loss", "mse"],
            { height: 200, callbacks: ["onEpochEnd"] }
        ),
    });
}


function testModel(model, inputData, normalizationData) {
    const { inputMax, inputMin, labelMax, labelMin } = normalizationData;

    const [xs, preds] = tf.tidy(() => {

        const xs = tf.linspace(0, 1, 100);

        const preds = model.predict(xs.reshape([100, 1]));

        const unNormsXs = xs.mul(inputMax.sub(inputMin)).add(inputMin);
        const unNormsPreds = preds.mul(labelMax.sub(labelMin)).add(labelMin);

        return [unNormsXs.dataSync(), unNormsPreds.dataSync()];

    });

    const predictedPoints = Array.from(xs).map((val, i) => {
        return { x: val, y: preds[i] };
    });


    const originalPoints = inputData.map((d) => {
        return {
            x: d.horsepower,
            y: d.mpg
        }
    });

    tfvis.render.scatterplot(
        { name: "Previsões vs Valores Originais"},
        {
            values: [originalPoints, predictedPoints],
            series: ["original","predicted"],
        },
        {
            xLabel: "HorsePower",
            yLabel: "MPG",
            height: 300
        }
    );

}

async function run() {
    const data = await getData();

    const values = data.map(d => ({
        x: d.horsepower,
        y: d.mpg
    }));

    tfvis.render.scatterplot(
        { name: "Horsepower vs MPG" },
        { values },
        {
            xLabel: "Horsepower",
            yLabel: "MPG",
            height: 300
        }
    );

    const tensorData = convertToTensor(data);
    const { inputs, labels } = tensorData;
    await trainModel(model, inputs, labels);
    console.log("Treinamento Completo");
    testModel(model,data,tensorData);


}

document.addEventListener('DOMContentLoaded', run);

const model = createModel();

tfvis.show.modelSummary({ name: "Modelo" }, model);

