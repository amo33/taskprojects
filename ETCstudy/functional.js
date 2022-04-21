
let answer = ''
for(let i=0; i<4; i++){
    answer += String(Math.floor(Math.random()* 10))
}
let expect = ''
while(answer !== expect){
    expect = ''
    let guess = 0 
    let idx = -1
    let strike = 0
    let ball = 0 
    /*
    for( let i=0; i<4; i++){
        guess=Math.floor(Math.random() * 10); // 0~9
        expect+=String(guess)
        */
    expect = prompt('값을 입력해주세요.');
       
    idx = expect.findIndex((data)=>{return data== guess})
    if(idx !== i){
        ball += 1
    }
    else if(idx === i){
        strike += 1
    }
    //} 
    console.log("ball",ball)
    console.log("Strike",strike)
}
