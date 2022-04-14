import React from "react";
import {Link} from "react-router-dom";
import Button from "@material-ui/core/Button";
function ShowData(props){
    
    if((props.method !== 'default')){
        return (
            <div>
                <div><h4>{props.method}</h4></div>
            <table>
                <thead>
                    <tr> 
                    <th>Name</th>
                    <th>Age</th>
                    {(props.status === 'deta') ? <th>user_id</th> : null}
                    {(props.status === 'list') ? <th>Image_Flag</th> : null}
                    {(props.status === 'detail') ? <th>Image path</th> : null}
                    </tr>
                </thead>
                <tbody>

                    {props.userdata.map((val, key) => {
                    return (
                        <tr key={key}>
                        {(props.status === 'list') ? <td> <Link to = {'/members/'+ val.id + '/'+ props.method}>{val.username}</Link></td> : null}
                        {(props.status === 'detail') ? <td>{val.username}</td> : null}
                        <td>{val.age}</td>
                        {(props.status === 'deta')? <td>{val.user_id}</td> : null}
                        {(props.status === 'list') ?<td>{val.Image_flag}</td>: null}
                        {(props.status === 'list') ? null :
                         (props.status === 'detail' && (val.image_path != '0' && val.image_path != undefined)) ? <td><img src= {process.env.PUBLIC_URL+val.image_path}/></td> : <td>No Image</td>}
                        
                        </tr>
            )}
            )}
                </tbody>
             </table>
             <div>
               <Button color = 'secondary' value={'showdefault'}>
               <Link to = '/'>Go to start page </Link>
               </Button>
            </div>
        </div>
        );
    
    }
    else{
        return null;
    }
}

export default ShowData;