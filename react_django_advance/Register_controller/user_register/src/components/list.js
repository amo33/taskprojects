import React, {useState} from "react";
import  Typography  from "@material-ui/core/Typography";
import Grid from "@material-ui/core/Grid";
import {Link,} from "react-router-dom";
import Button from "@material-ui/core/Button";
import ShowData from "./table.js";
import axios from "axios";

const List=()=>{
    const [status, setstatus] = useState('default');
    const [data, setdata] = useState([]);

    const handledataupdate = (updata) =>{ //data 업데이트
        setdata([...updata]);
    }
    
    const onhandlestatus = (state)=>{ // 목록페이지 중 txt or db 중으로 볼지 업데이트 
        setstatus(state);
    }
    
    const handleToseedata=(val)=>{ //tsv or db 보고 싶으면.
        axios.get('api/users'+'?method='+val)
      .then(response => {
          handledataupdate(response.data);
          onhandlestatus(val);
        })
      .catch((Error)=>{alert(Error)});
    };
    const defaultpage=()=>{
        return (
            <div>
                <div id= "default">
                <Grid container spacing = {2}>
                    <Grid item xs = {8} align= 'center'>
                        <Typography component = "h4" variant = "h4" style={{cursor:'default'}}>
                            Choose between db and text 
                        </Typography>
                    </Grid>
                    <Grid item xs={8} align="center">
                        <Button color= 'primary' variant="contained" value= {'db'} onClick={() => handleToseedata('db')}>
                            <Link to = '/list?category=showdb'>show me Database</Link>
                        </Button>
                    </Grid>
                    <Grid item xs={8} align="center">
                        
                        <Button color= 'secondary' variant="contained" value={'text'} onClick={() => handleToseedata('text')}>
                        <Link to = '/list?category=showlist'>show me text </Link>
                        </Button>
                        
                    </Grid>
                </Grid>
                
                </div>
                <div>
                    <Button color = 'secondary' value={'showdefault'}>
                    <Link to = '/'>Go to start page </Link>
                    </Button>
                </div>
             </div>
        );
    }   

    return (status === 'default' ? defaultpage() : <div id= "showingData"><ShowData userdata = {data} method = {status} status = "list"/></div>);
        
}

export default List