import React from "react";
import {
    BrowserRouter as Router, Route,
} from "react-router-dom";
import Header from "./Header"
import Home from "./Home";
import list from "./list";


export default function Routes() {
    return (
        <Router>
            <Header /> 
            <Route path="/" component={Home} /> 
            <Route path="/list" component={list} />
        </Router>
    )
}