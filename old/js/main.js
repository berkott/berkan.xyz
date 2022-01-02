$(document).ready(function() {
	$('#pagepiling').pagepiling({
        anchors: ['page1', 'page2', 'page3', 'page4', 'page5', 'page6', 'page7', 'page8', 'page9',
            'page10', 'page11', 'page12', 'page13'],
        sectionsColor: ['#314776', '#5b5b5b', '#d28a22', '#0f1745', '#19356c', '#327250', '#1b3596',
            '#7b572f', '#485013', '#2A4396', '#a22729', '#111111'],
        navigation: { //#7aa1ff
            'position': 'right',
            'tooltips': ['Home', 'Covid Detection Kaggle', 'Dev Fruit', 'Blind Assistant', 'Techno Titans Robotics',
                'Sleep Apnea AI','Destination: Titan', 'Mole Project', 'Space Invaders AI', 'SmartLib', 
                'Techno Titans', 'The Game Palace']
        },

        afterRender: function(){
            $('#pp-nav').addClass('custom');
        },
        afterLoad: function(anchorLink, index){
            if(index>1){
                $('#pp-nav').removeClass('custom');
            }else{
                $('#pp-nav').addClass('custom');
            }
        }
    });
});
