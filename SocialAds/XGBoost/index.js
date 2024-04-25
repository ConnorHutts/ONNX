// Function to run inference
async function runExample() {
    // Load the ONNX model
    const session = await ort.InferenceSession.create();
    await session.loadModel('./xgboost_Ads_ort.onnx');

    // Retrieve input values from the HTML input elements
    const x1 = parseFloat(document.getElementById('box1').value);
    const x2 = parseFloat(document.getElementById('box2').value);

    // Prepare input tensor for inference
    const inputTensor = new ort.Tensor(new Float32Array([x1, x2]), [1, 2]);

    try {
        // Run inference with input tensor
        const output = await session.run({ 'X': inputTensor });

        // Display inference result
        const prediction = output['Y'].data[0]; // Assuming 'Y' is the output node name
        const predictionsElement = document.getElementById('predictions');
        predictionsElement.innerHTML = `<p>Model prediction: ${prediction}</p>`;
    } catch (error) {
        console.error('Error running inference:', error);
    }
}
