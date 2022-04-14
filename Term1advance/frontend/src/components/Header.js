import React from "react";
import { useState } from "react";
import axios from "axios";
//yarn start 로 실행하셈
function Header(){
  const [inputs, setInputs] = useState({
      name: '',
      age : 0,
      image : ''
  });
  const {name, age, image} = inputs;

  const onChange = (e) => {
      const {value, name} = e.target;
      const {ageval, age} = e.target;
      // const {file, image} = e.target.files; // 이거 사용할시 name, age 입력이 안된다.
    setInputs({
        ...inputs,
        [name]:value,
        [age]:ageval,
        //[image]:file,
    });
  };
  const handleClick = (e)=>{
      const formdata = new FormData();
      formdata.append('uploadImage', image[0]);

      const config = {
          Headers: {
              'content-type':'multipart/form-data'
          },
      };
      axios.post('api',formdata, config)
  }
  const onReset = () => {
    setInputs({
        name: '',
        age: 0,
        image: ''
    });
  };

  return (
    <div>
      <input name="name" placeholder="이름" onChange={onChange} value={name} />
      <input name="age" placeholder="나이" onChange={onChange} value={age} />
      <input type="file" accept="img/jpg,image/png,image/jpg,image/gif" onChange={onChange} />
      <button onClick={onReset}>초기화</button>
      <div>
        <b>값: {name} ({age})</b>
    
      </div>
    </div>
  );
}

export default Header;