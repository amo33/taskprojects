### 1.자바스크립트 함수형 코딩 
        입출력이 순수해야한다. (순수함수)
        부작용(부산물)이 없어야한다.
        함수와 데이터를 중점으로 생각.
    -받은 인자암으로 결과물을 내어야하는 함수 : 순수함수 :: 따라서 자바스크립트의 this 때문에 완벽한 함수형 프로그래밍은 힘들다. 
    프로그래머가 바꾸고자하는 변수 외에는 아무것도 바뀌어서는 안된다. 원본 데이터(함수)는 불변이어야한다. 

    ```javascript 
            // 대표적인 함수 : map, filter, reduce  
            var input = [1, 2, 3, 4, 5];
            var res = input.map(function(x) {
            return x * 2;
            }); // [2, 4, 6, 8, 10]
    ```
    위에서 본 map 함수에서도 입력값 arr에 대해서 입력값은 변하지 않고, map이라는 결과도 입력에 관해서만 나왔다. 

    ``` javascript

        var input = [1, 2, 3, 4, 5];
        var val = function(x) { return x % 2 === 0; }
        var check = function(input) {
        return input.filter(val);
        };
        check(input); // [2, 4] 
    ```
    위에 있는 이 함수는 check은 순수 함수 일까? : val이라는 입력받지 않은 변수를 사용했으므로 순수함수가 아니다.

    ```javascript

        var input = [1, 2, 3, 4, 5];
        var val = function(x) { return x % 2 === 0; }
        var check = function(input, val) {
        return input.filter(val);
        };
        check(input, val); // [2, 4] 
    ```
    이런 식으로 변수 또한 인자로 받으면 check도 순수함수가 된다. 