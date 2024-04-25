async function runExample() {

    var x = new Float32Array( 1, 2 )

    var x = [];

     x[0] = document.getElementById('box1').value;
     x[10] = document.getElementById('box2').value;

    let tensorX = new ort.Tensor('float32', x, [1, 11] );
    let feeds = {float_input: tensorX};

    let session = await ort.InferenceSession.create('xgboost_Ads_ort.onnx');
    
   let result = await session.run(feeds);
   let outputData = result.variable.data;

  outputData = parseFloat(outputData).toFixed(2)

   let predictions = document.getElementById('predictions');

  predictions.innerHTML = ` <hr> Got an output tensor with values: <br/>
   <table>
     <tr>
       <td>  Ad Purchases  </td>
       <td id="td0">  ${outputData}  </td>
     </tr>
  </table>`;
    


}
