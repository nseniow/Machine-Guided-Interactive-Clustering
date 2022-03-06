import React from 'react';
import {Card} from 'react-bootstrap'

export const ChartSlot = (props) => {
    return (
        <div>
            <Card className="bg-dark">
                <Card.Img src={props.imgSrc} alt="Card image"/>
                <Card.ImgOverlay className="lessImagePadding text-center">
                </Card.ImgOverlay>
            </Card>
        </div>
    );
};