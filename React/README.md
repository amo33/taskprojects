#### React native study 

1. We can't use html and css. So we need to use jsx (javascript xml)
```jsx
    const name = 'Josh Perez';
    const element = <h1>Hello, {name}</h1>;
    // 이런식으로 변수에 접근하려면 {}를 써야한다.
    function formatName(user) {
    return user.firstName + ' ' + user.lastName;
    }

    const user = {
    firstName: 'Harper',
    lastName: 'Perez'
    };

    const element = (
    <h1>
        Hello, {formatName(user)}!
    </h1>
    );
    // 마찬가지로 함수 결과도 {}로 접근한다.
    // 변수를 html의 h1 tag나 다른 곳에 넣어주려면 {} 를 쓰는거고, 조건문 같은 곳에 쓰인다면 {}는 안쓴다.

    const element = React.createElement(
    'h1',
    {className: 'greeting'},
    'Hello, world!'
    ); // 이렇듯 하나를 생성하고 할당하는 방식도 있다.
```
2. <div id="root"></div> - 이 ReactDom 노드인데, 이 안에 들어가는 모든 element를 React Dom에서 관리해준다. 
```javascript
    const element = <h1>Hello, world</h1>
    ReactDom.render(element, document.getElementbyId('root')) 
    // 렌더링 방식이다. 이러면 hello world가 보인다.
```