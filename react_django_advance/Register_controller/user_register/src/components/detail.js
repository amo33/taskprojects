import React from "react";

function ShowDetail(props){

    if(props.state !== 0 && props.state!== undefined){
        return (
            <div>
                <h4>{props.method}</h4>
            <table>
                <thead>
                    <tr>
                    <th>Name</th>
                    <th>Age</th>
                    <th>User_id</th>
                    <th>Image</th>
                    </tr>
                </thead>
                <tbody>

                    {props.data.map((val, key) => {
                    return (
                        <tr key={key}>
                        <td>{val.username}</td>
                        <td>{val.age}</td>
                        <td>{val.user_id}</td>
                        <td><img src= {process.env.PUBLIC_URL+val.image_path} alt="No Image Registered"/></td>
                        </tr>
        )
        })}
        </tbody>
    </table>
    </div>
        )
    }
}

export default ShowDetail;