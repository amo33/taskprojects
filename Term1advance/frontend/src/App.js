import './App.css';
import React, {Component} from 'react';
import axios from 'axios';
import {Link} from "react-router-dom"
import {
  BrowserRouter as Router, Route,
} from "react-router-dom";
import Header from './components/Header';
/*
function textInput() {
  var getTitle = document.getElementById("title").value;
  var getContent = document.getElementById("content").value;
  axios.post("http://127.0.0.1:8000/api/", {
    title: getTitle,
    content: getContent,
  })
    .then(function (response) {
      console.log(response);
    })
    .catch(function (error) {
      console.log(error);
    });
  <Link to="/notice"></Link>
}
<Link to="/notice"> <button type="button" className="post" onClick={textInput}>
  등록하기
</button> </Link>
*/
/*
class App extends Component {
  render(){
    const users = [
      { name: 'Nathan', age: 25 },
      { name: 'Jack', age: 30 },
      { name: 'Joe', age: 28 },
    ];

    return (
      <div>
      <ol>
        {users
          .map(user => <li>{user.name}</li>)
        }
      </ol>
      <header_ex />
      </div>
    );
  }
}
*/
function App(){
  return (
    <div>
    <h1>Submit list </h1>
    <Header />
    </div>
  );
}
export default App;
